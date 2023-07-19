import itertools
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class DBLP:
    def __init__(self, graph = nx.Graph(), dataset: str = ""):
        self.graph: nx.Graph = graph
        
        # Dictionaries to store node numbers and corresponding IDs
        self.data_id_to_node_id: dict = {} # Dictionary linking pubblication or author to relative node number
        self.node_id_to_data_id: dict = {} # Dictionary linking node number to relative pubblication or author

        self.dataset: str = dataset # dataset name
    

    # Function to plot graph
    def plot_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos=pos, node_color="red", font_color="white")

        labels = {node : (node, self.graph.nodes[node]['bipartite']) for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos=pos, labels=labels)

        plt.margins(0.2)
        plt.show()

    # Function to add a node to the graph and update dictionaries
    def add_node_to_graph_bipartite(self, id: str, isPublication: bool, labels: dict = None):
        if id not in self.data_id_to_node_id:
            node = len(self.data_id_to_node_id) # Get node number
            
            # Key Data id (author name or publication id) Value node id
            self.data_id_to_node_id[id] = node

            # Key Node id (number) Value author name or publication id
            self.node_id_to_data_id[node] = id

            if isPublication:
                self.graph.add_node(node, bipartite=0, labels=labels) # Add node of publication
            else:
                self.graph.add_node(node, bipartite=1, labels=labels) # Add node of author

    # Function to create a bipartite graph
    def create_graph_bipartite(self, data: pd.DataFrame):
        for row in tqdm(data.itertuples(index=False), desc="Create graph"):
            publication = row.id
            
            # Set labels for publication nodes
            labels = {
                    "year": row.year.split('|')[0], 
                    "title": row.title, 
                    "pages": row.pages if self.dataset != "mastersthesis" else None,
                    }
            
            if self.dataset == "inproceedings":
                labels["publisher"] = row.editor
            elif self.dataset == "mastersthesis":
                labels["publisher"] = row.school
            else:
                labels["publisher"] = row.publisher
                
            if self.dataset == "article":
                labels["venue"] = row.journal
            elif self.dataset == "incollection" or self.dataset == "inproceedings":
                labels["venue"] = row.booktitle
            elif self.dataset == "proceedings":
                labels["venue"] = row.title
            
            # Add publication node to graph
            self.add_node_to_graph_bipartite(id=publication, isPublication=True, labels=labels)

            authors: str = row.author.split('|')
            for author in authors:
                # Add author node to graph
                self.add_node_to_graph_bipartite(id=author, isPublication=False, labels=None)

                # Add unidrected edge between author and publication
                self.graph.add_edge(self.data_id_to_node_id[author], self.data_id_to_node_id[publication])
        
        print(f"Graph created! {self.graph}")
    

    #def add_node_to_graph(self, id: str):


    # Function to return author(s) who wrote more papers by himself within a certain year 
    def ex_1(self, threshold_year: int):
        authors_count = {} # Dictionary for counting number of publications an author has written by himself

        publications = [node for node in self.graph.nodes if self.graph.nodes[node]["bipartite"] == 0]
        for publication in tqdm(publications, desc="Finding author with most publication written by himself"):
            # Get publications written only by one author
            if (int(self.graph.nodes[publication]["labels"]["year"]) <= threshold_year # Check if article is written before threshold year
                and self.graph.degree[publication] == 1 # Check if node has only one author (i.e. one indegree)
            ):
                # Get node of author who written the publication by himself
                author_node_id = list(self.graph[publication])[0]
                # Add author to dict or increment count of publications who written by himself
                authors_count[author_node_id] = authors_count.get(author_node_id, 0) + 1

        # Get the maximum number of publications written by a single author
        max_publications_count = max(authors_count.values(), default=0)
        
        # Return all author who written the maximum number of publications by himself
        return [(self.node_id_to_data_id[author_node_id], count) for author_node_id, count in authors_count.items() if count == max_publications_count]

    # Function to return exact diameter of graph
    def ex_2(self, threshold_year: int):
        # Remove the pubblication with year largest than threshold
        publications_to_remove = [node for node in self.graph.nodes
                                  if self.graph.nodes[node]["bipartite"] == 0
                                  and int(self.graph.nodes[node]["labels"]["year"]) > threshold_year]
        filtered_graph: nx.Graph = self.graph.copy()
        filtered_graph.remove_nodes_from(publications_to_remove)

        # Find the largest connected component graph
        largest_connected_component_graph: nx.Graph = nx.subgraph(self.graph, max(nx.connected_components(filtered_graph), key=len, default=0))

        # Get node with maximum degree to start algorithm
        start_node = max(nx.degree(largest_connected_component_graph), key = lambda x : x[1])[0]

        # Set lower bound and upper bound
        level = nx.eccentricity(largest_connected_component_graph, v=start_node)
        lower_bound = level
        upper_bound = 2 * level

        while upper_bound > lower_bound:
            # Get all nodes at distance i from u
            nodes_distance_i = [node for node, distance in nx.shortest_path_length(largest_connected_component_graph, source=start_node).items() 
                                        if distance == level]

            # Get maximum eccentrity from u in F_i (i.e. B_i)
            max_eccentrity_distance_i = max([nx.eccentricity(largest_connected_component_graph, v=node) for node in nodes_distance_i])

            if max(lower_bound, max_eccentrity_distance_i) > 2 * (level - 1):
                return max(lower_bound, max_eccentrity_distance_i)
            else:
                lower_bound = max(lower_bound, max_eccentrity_distance_i)
                upper_bound = 2 * (level - 1)
            level = level - 1
            print(f"Lower bound: {lower_bound} - Upper bound: {upper_bound}")
        return lower_bound
        
    # Function to return pair of publications shared maximum number of authors
    def ex_3(self, threshold_year: int):
        # Get all authors nodes
        authors = [node for node in self.graph.nodes if self.graph.nodes[node]["bipartite"] == 1]

        # Dictionary to count occurence of pairs
        pair_counts = {}
        for author in tqdm(authors, desc="Find publications pair sharing max number of authors"):
            # Get all publications written by an author
            author_publications = [node for node in self.graph[author] if int(self.graph.nodes[node]["labels"]["year"]) <= threshold_year]

            # Get all pairs combination of publications written by an author
            publication_pairs = itertools.combinations(author_publications, r=2)
            # Count occurrence of publication pair
            for pair in publication_pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        
        # Get pair of publications who shared maximum number of authors (i.e. pairs with maximum count)
        max_shared_pair = max(pair_counts, key=pair_counts.get, default=None)

        return  max_shared_pair

    
def main():
    path = './DBLP/'
    files_name = ['out-dblp_article', 'out-dblp_book', 'out-dblp_incollection', 'out-dblp_inproceedings', 'out-dblp_mastersthesis', 'out-dblp_phdthesis', 'out-dblp_proceedings']
    datasets_name = ['article', 'book', 'incollection', 'inproceedings', 'mastersthesis', 'phdthesis', 'proceedings']
    dblp_list = [DBLP(graph=nx.Graph(), dataset=dataset_name) for _, dataset_name in zip(files_name, datasets_name)]
    
    nrows = 40
    for i, file_name in enumerate(files_name):
        print(f"{file_name}\n\n") 
        # Read csv file and create pandas Serie
        print("Loading data...")
        data = pd.read_csv(f'{path + file_name}.csv', sep=';', dtype=str, nrows=nrows)

        # Remove rows with NaN values in the 'author' field
        data.dropna(subset=['author'], inplace=True)

        dblp_list[i].create_graph_bipartite(data)

        '''
        # Exercise 1
        threshold_year = 2020
        max_himself = dblp[i].ex_1(threshold_year=threshold_year)
        if max_himself:
            [print(f"{author}: {count}") for author, count in max_himself]
        else:
            print(f"No author has written a publication by himself until {threshold_year}!")

        # Exercise 2
        threshold_year = 2020
        diameter = dblp[i].ex_2(threshold_year=threshold_year)
        print(f"Diameter: {diameter}")

        # Exercise 3
        threshold_year = 2020
        max_shared_publications = dblp[i].ex_3(threshold_year=threshold_year)
        print(f"Coppia di articoli che condivide il maggior numero di autori: {max_shared_publications}")
        '''
        print("--------------------------------------------")

    # Crate union graph renaming nodes Graph.clear()
    union_graph: nx.Graph() = nx.union_all([dblp_dataset.graph for dblp_dataset in dblp_list], datasets_name)
    union_dblp = DBLP(graph=union_graph)
    print(f"Union graph created! {union_dblp.graph}")

    # Merge dictionaries in union graph 
    for dblp_dataset, dataset_name in zip(dblp_list, datasets_name):

        # clear graph to free ram memory space
        dblp_dataset.graph.clear()
        
        # replace the old nodes id to new renamed nodes id
        # for example mastersthesis graph nodes id are renamed as mastersthesisx, for each x belongs to N
        mappings_from_old_node_id_to_new_node_id = {}
        for node_id in dblp_dataset.node_id_to_data_id:
            mappings_from_old_node_id_to_new_node_id[node_id] = dataset_name + str(node_id)
        dblp_dataset.node_id_to_data_id = dict((mappings_from_old_node_id_to_new_node_id[node_id], value) for (node_id, value) in dblp_dataset.node_id_to_data_id.items())
    
    # Updates the node_id_to_data_id dictionary with the elements from every dblp_dataset node_id_to_data_id dictionary object
    for dblp_dataset in dblp_list:
        union_dblp.node_id_to_data_id.update(dblp_dataset.node_id_to_data_id)


    '''
    a = {'a' : 0, 'b': 1}
    b = {'c' : 2, 'b': 4}
    for i in a:
        i = a["article" + str(i)]
    b.update(a)
    print(b)
    
    
    for node in union_graph.nodes:
        # article
        if union_graph.nodes[node]['bipartite'] == 0:
            if union_graph.nodes[node]['labels']['id'] is not union_dblp.id_to_node:
                union_dblp.id_to_node[union_graph.nodes[node]['labels']['id']] = node
            # duplicate
            else:
                union_dblp.id_to_node["qualcosa"+union_graph.nodes[node]['labels']['id']] = node
        # author
        else:
            if union_graph.nodes[node]['labels']['id'] is not union_dblp.id_to_node:
                union_dblp.id_to_node[union_graph.nodes[node]['labels']['id']] = node
            else:
    union_dblp.id_to_node = 
    
        a = {'a' : 0, 'b' : 1}
        b = {'c' : 2, 'b':4}

        a.update(b)
        print(a)
        {'a': 0, 'b': 4, 'c': 2}
    

    a = {'a' : 0, 'b': 1}
    b = {'c' : 2, 'b': 4}
    for i in a:
        a[i] = "article" + str(a[i])
        a.update(b)
    print(a)



    for dblp_dataset in dblp:
        for key in dblp_dataset.id_to_node:
            dblp_dataset.id_to_node[key] +
        
        union_dblp.id_to_node.update(dblp_dataset.id_to_node)
    '''
    # Exercise 1
    threshold_year = 2020
    max_himself = union_dblp.ex_1(threshold_year=threshold_year)
    if max_himself:
        [print(f"{author}: {count}") for author, count in max_himself]
    else:
        print(f"No author has written an publication by himself until {threshold_year}!")

    # Exercise 2
    threshold_year = 2020
    diameter = union_dblp.ex_2(threshold_year=threshold_year)
    print(f"Diameter: {diameter}")

    # Exercise 3
    threshold_year = 2020
    max_shared_publications = union_dblp.ex_3(threshold_year=threshold_year)
    print(f"Coppia di articoli che condivide il maggior numero di autori: {max_shared_publications}")

if __name__ == "__main__":
    main()


# TODO vedere se la 3 si puo fare meglio evitando di memorizzare tutto (inoltre vedere anche se ci più coppie che condividono lo stesso numero max)
# TODO esercizio 4
# TODO riscrivere meglio le variabili i commenti e i print


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