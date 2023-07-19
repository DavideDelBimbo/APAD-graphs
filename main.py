import itertools
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class DBLP:
    def __init__(self):
        self.graph: nx.Graph = nx.Graph()
        
        # Dictionaries to store node numbers and corresponding IDs
        self.id_to_node: dict = {} # Dictionary linking pubblication or author to relative node number
        self.node_to_id: dict = {} # Dictionary linking node number to relative pubblication or author
    

    # Function to plot graph
    def plot_graph(self):
        #l, r = nx.bipartite.sets(self.graph)
        #pos = nx.bipartite_layout(self.graph, l)
        #nx.draw(self.graph, pos=pos, with_labels=True, )

        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos=pos, node_color="red", font_color="white")

        labels = {node : (node, self.graph.nodes[node]['bipartite']) for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos=pos, labels=labels)

        plt.margins(0.2)
        plt.show()

    # Function to add a node to the graph and update dictionaries
    def add_node_to_graph(self, id: str, isArticle: bool, labels: dict = None):
        if id not in self.id_to_node:
            node = len(self.id_to_node) # Get node number
            
            self.id_to_node[id] = node
            self.node_to_id[node] = id

            if isArticle:
                self.graph.add_node(node, bipartite=0, labels=labels) # Add node of article
            else:
                self.graph.add_node(node, bipartite=1, labels=labels) # Add node of author

    # Function to create a bipartite graph
    def create_graph(self, data: pd.DataFrame):
        for row in tqdm(data.itertuples(index=False), desc="Create graph"):
            article = row.id
            labels = {
                    "year": row.year, 
                    "title": row.title, 
                    "pages": row.pages, 
                    "publisher": row.publisher,
                    "venue": row.journal
                    }
            # Add article node to graph
            self.add_node_to_graph(id=article, isArticle=True, labels=labels)

            authors: str = row.author.split('|')
            for author in authors:
                # Add author node to graph
                self.add_node_to_graph(id=author, isArticle=False)

                # Add unidrected edge between author and article
                self.graph.add_edge(self.id_to_node[author], self.id_to_node[article])
        
        print(f"Graph created! {self.graph}")
    
    # Function to return author(s) who wrote more papers by himself within a certain year 
    def ex_1(self, threshold_year: int):
        authors_count = {} # Dictionary for counting number of articles an author has written by himself

        for node in tqdm(self.graph.nodes(), desc="Finding author with most article written by himself"):
            # Get articles written only by one author
            if (self.graph.nodes[node]["bipartite"] == 0 # Check if node is relative to article
                and int(self.graph.nodes[node]["labels"]["year"]) <= threshold_year # Check if article is written before threshold year
                and self.graph.degree[node] == 1 # Check if node has only one author (i.e. one indegree)
            ):
                # Get node of author who written the article by himself
                author = [n for n in self.graph.neighbors(node)][0]
                # Add author to dict or increment count of articles who written by himself
                if author not in authors_count:
                    authors_count[author] = 1
                else:
                    authors_count[author] += 1

        # Get the maximum number of articles written by a single author
        if len(authors_count) > 0:
            max_articles = max(authors_count.values())
            # Return all author who written the maximum number of articles by himself
            return [(self.node_to_id[author], count) for author, count in authors_count.items() if count == max_articles]
        else:
            return None

    # Function to return exact diameter of graph
    def ex_2(self, threshold_year: int):
        # Find the largest connected component
        graph_largest_component = nx.subgraph(self.graph, max(nx.connected_components(self.graph), key=len))

        # Get node with maximum degree to start algorithm
        u_node_maximum_degree = max(nx.degree(graph_largest_component), key = lambda x : x[1])[0]

        # Set lower bound and upper bound O(|V| + |E|)
        i_level = nx.eccentricity(graph_largest_component, v=u_node_maximum_degree)
        lower_bound = i_level
        upper_bound = 2*i_level

        while upper_bound > lower_bound:
            # Get all nodes at distance i from u
            F_nodes_distance_i_from_u = [node for node,
                                          distance in nx.shortest_path_length(graph_largest_component,
                                          source=u_node_maximum_degree, method='dijkstra').items() 
                                        if distance == i_level]

            #Get maximum eccentrity from u in F_i
            B_max_eccentrity_from_F_i = max([nx.eccentricity(graph_largest_component, v=node) for node in F_nodes_distance_i_from_u])

            if max(lower_bound, B_max_eccentrity_from_F_i) > 2 * (i_level - 1):
                return max(lower_bound, B_max_eccentrity_from_F_i)
            else:
                lower_bound = max(lower_bound, B_max_eccentrity_from_F_i)
                upper_bound = 2 * (i_level - 1)
            i_level = i_level-1
            print(f"Lower bound: {lower_bound} - Upper bound: {upper_bound}")
        return lower_bound
        
    # Function to return pair of articles shared maximum number of authors
    def ex_3(self, threshold_year: int):
        # Get all authors nodes
        authors = [node for node in self.graph.nodes if self.graph.nodes[node]["bipartite"] == 1]

        # Dictionary to count occurence of pairs
        pair_counts = {}
        for author in tqdm(authors, desc="Find articles pair sharing max number of authors"):
            # Get all articles written by an author
            author_articles = list(self.graph[author])
            # Get all pairs combination of articles written by an author
            article_pairs = itertools.combinations(author_articles, r=2)
            # Count occurrence of article pair
            for pair in article_pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        # Get pair of articles who shared maximum number of authors (i.e. pairs with maximum count)
        max_shared_pair = max(pair_counts, key=pair_counts.get)
        max_shared_value = pair_counts[max_shared_pair]

        return max_shared_pair, max_shared_value


def main():
    path = './DBLP/'
    files_name = ['out-dblp_article', 'out-dblp_book', 'out-dblp_incollection', 'out-dblp_inproceedings', 'out-dblp_mastersthesis', 'out-dblp_phdthesis', 'out-dblp_proceedings']

    # Read csv file and create pandas Serie
    nrows = None
    print("Loading data...")
    data = pd.read_csv(f'{path + files_name[0]}.csv', sep=';', dtype=str, nrows=nrows)

    # Remove rows with NaN values in the 'author' field
    data.dropna(subset=['author'], inplace=True)
    
    dblp = DBLP()
    print("Creating graph...")
    dblp.create_graph(data)
    '''
    # Exercise 1
    threshold_year = 2020
    max_himself = dblp.ex_1(threshold_year=threshold_year)
    if max_himself:
        [print(f"{author}: {count}") for author, count in max_himself]
    else:
        print(f"No author has written an article by himself until {threshold_year}!")

    # Exercise 2
    threshold_year = 2020
    diameter = dblp.ex_2(threshold_year=threshold_year)
    print(f"Diameter: {diameter}")
    '''
    # Exercise 3
    threshold_year = 2020
    max_shared_articles, max_shared_authors = dblp.ex_3(threshold_year=threshold_year)
    print(f"Coppia di articoli che condivide il maggior numero di autori: {max_shared_articles} - {max_shared_authors}")
    print(dblp.graph.nodes[max_shared_articles[0]], dblp.graph.nodes[max_shared_articles[1]])
if __name__ == "__main__":
    main()


# TODO inserire year_threshold in ogni esercizio
# TODO vedere se la 3 si puo fare meglio evitando di memorizzare tutto (inoltre vedere anche se ci più coppie che condividono lo stesso numero max)
# TODO esercizio 4
# TODO riscrivere meglio le variabili i commenti e i print
# TODO implementare una soluzione per eseguire su tutti e 7 i dataset


'''
Fields id;author;author-aux;author-orcid;booktitle;cdate;cdrom;cite;cite-label;crossref;editor;editor-orcid;ee;ee-type;i;journal;key;mdate;month;note;note-label;note-type;number;pages;publisher;publnr;publtype;sub;sup;title;title-bibtex;tt;url;volume;year
4105295;Clement T. Yu|Hai He|Weiyi Meng|Yiyao Lu|Zonghuan Wu;;;;;;;;;;;https://doi.org/10.1007/s11280-006-0010-9;;;World Wide Web;conf/www/HeMLYW07;2017-05-23;;;;;2;133-155;;;;;;Towards Deeper Understanding of the Search Interfaces of the Deep Web.;;;db/journals/www/www10.html#HeMLYW07;;2007
'''


'''
Note that the field author contains the list of the authors of the publications separated by "|"

For each of the 7 dataset build the bipartite graph authors vs publication using Networkx. 
In this graph each publication is a node and also each author correspond to a node. 
There is an edge (undirected) between an author and a publication if the author authored the publication.


Tip: each publication and each author will have a number in the graph.
Build a dictionary associating to each publication or author the number of the corresponding node in the graph.
Also, build the reverse dictionary (or store as an attribute on the nodes) which says for each node number the corresponding publication id or author.
'''

'''
it is convenient to put a label on each publication, which tells its year, title, number of pages, publisher, venue.
For venue we mean: 
journal (for out-dblp_article.csv), 
booktitle (for out-dblp_incollection.csv and out-dblp_inproceedings.csv),
title (for out-dblp_proceedings.csv)
'''

'''
For each graph, considering only the publications up to year x with x in {1960,1970,1980,1990,2000,2010,2020,2023}:
QUESTION 1:	L- Who is the author who wrote more papers by himself/herself?

QUESTION 2:	2-Compute exactly the diameter of G

QUESTION 3:	III-Which is the pair of papers that share the largest number of authors?

QUESTION 4: Build the union graph and repeat the chosen questions.
Build also the author graph, whose nodes are only authors and two authors are connected if they did a publication together (considering all the files). Answer to the following question:
Which is the pair of authors who collaborated the most between themselves?
'''