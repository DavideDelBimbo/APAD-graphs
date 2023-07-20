import itertools
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

'''
QUESTION 1:	L-Who is the author who wrote more papers by himself/herself?

QUESTION 2:	2-Compute exactly the diameter of G

QUESTION 3:	III-Which is the pair of papers that share the largest number of authors?

QUESTION 4: Build the union graph and repeat the chosen questions.
Build also the author graph, whose nodes are only authors and two authors are connected if they did a publication together (considering all the files). Answer to the following question:
Which is the pair of authors who collaborated the most between themselves?
'''

class DBLP:
    def __init__(self, graph = nx.Graph(), dataset_name: str = ""):
        self.graph: nx.Graph = graph
        
        # Dictionaries to store node numbers and corresponding IDs
        self.data_id_to_node_id: dict = {} # Dictionary linking pubblication or author to relative node number
        self.node_id_to_data_id: dict = {} # Dictionary linking node number to relative pubblication or author

        self.dataset_name: str = dataset_name # dataset name
    

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
        start = time.time()
        for row in tqdm(data.itertuples(index=False), desc="Create graph"):
            publication = row.id
            
            # Set labels for publication nodes
            labels = {
                    "year": row.year.split('|')[0], 
                    "title": row.title, 
                    "pages": row.pages if self.dataset_name != "mastersthesis" else None,
                    }
            
            if self.dataset_name == "inproceedings":
                labels["publisher"] = row.editor
            elif self.dataset_name == "mastersthesis":
                labels["publisher"] = row.school
            else:
                labels["publisher"] = row.publisher
                
            if self.dataset_name == "article":
                labels["venue"] = row.journal
            elif self.dataset_name == "incollection" or self.dataset_name == "inproceedings":
                labels["venue"] = row.booktitle
            elif self.dataset_name == "proceedings":
                labels["venue"] = row.title
            
            # Add publication node to graph
            self.add_node_to_graph_bipartite(id=publication, isPublication=True, labels=labels)

            authors: str = row.author.split('|')
            for author in authors:
                # Add author node to graph
                self.add_node_to_graph_bipartite(id=author, isPublication=False, labels=None)

                # Add unidrected edge between author and publication
                self.graph.add_edge(self.data_id_to_node_id[author], self.data_id_to_node_id[publication])
        
        end = time.time()
        print(f"Graph created in {end-start} seconds! {self.graph}")


    # Function to return author(s) who wrote more papers by himself within a certain year 
    def ex_1(self, threshold_year: int):
        start = time.time()
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

        end = time.time()
        print(f"Results calculated in {end-start} seconds")

        # Return all author who written the maximum number of publications by himself
        return [(self.node_id_to_data_id[author_node_id], count) for author_node_id, count in authors_count.items() if count == max_publications_count]

    # Function to return exact diameter of graph
    def ex_2(self, threshold_year: int):
        start = time.time()

        # Remove the pubblication with year largest than threshold
        publications_to_remove = [node for node in self.graph.nodes
                                  if self.graph.nodes[node]["bipartite"] == 0
                                  and int(self.graph.nodes[node]["labels"]["year"]) > threshold_year]
        filtered_graph: nx.Graph = self.graph.copy()
        filtered_graph.remove_nodes_from(publications_to_remove)

        # Find the largest connected component graph
        largest_connected_component_graph: nx.Graph = nx.subgraph(self.graph, max(nx.connected_components(filtered_graph), key=len, default=None))

        #print(f"Networkx diameter: {nx.approximation.diameter(largest_connected_component_graph)}\n")
        # Get node with maximum degree to start algorithm (if not exists the largest connected componento graph return diameter 0)
        start_node = max(nx.degree(largest_connected_component_graph), key = lambda x : x[1], default=None)
        if start_node:
            start_node = start_node[0] 
        else:
            return 0

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

        end = time.time()
        print(f"Results calculated in {end-start} seconds")
        return lower_bound
        
    # Function to return pair of publications shared maximum number of authors
    def ex_3(self, threshold_year: int):
        start = time.time()

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
        max_shared_value = max(pair_counts.values(), default=0)
        max_shared_pairs = [(self.node_id_to_data_id[pair[0]], self.node_id_to_data_id[pair[1]]) for pair, count in pair_counts.items() if count == max_shared_value]

        end = time.time()
        print(f"Results calcualted in {end-start} seconds")

        return [(max_shared_pair, max_shared_value) for max_shared_pair in max_shared_pairs]
    
    def ex_4(self):
        start = time.time()

        # Get all articles nodes
        articles = [node for node in self.graph.nodes if self.graph.nodes[node]["bipartite"] == 0]
        
        # Create the author graph
        author_graph: nx.Graph = nx.Graph()

        pair_counts = {}
        for article in tqdm(articles, desc="Create author graph and find pair with most collaborations"):
            # Getting neighbors authors that collaborates
            neighbors_authors = list(self.graph[article])

            # Add authors to graph
            author_graph.add_nodes_from(neighbors_authors)

            # Get combination of authors that collaborates
            neighbors_author_pairs = list(itertools.combinations(neighbors_authors, r=2))

            # Add edges to graph between authors that collaborates
            author_graph.add_edges_from(neighbors_author_pairs)

            # Count number of times a pair of authors that collaborates
            for pair in neighbors_author_pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        print(f"Author graph created! {author_graph}\n")

        # Get pair of authors that most collaborates (i.e. pairs with maximum count)
        max_collaboration_value = max(pair_counts.values(), default=0)
        max_collaboration_pairs = [(self.node_id_to_data_id[pair[0]], self.node_id_to_data_id[pair[1]]) for pair, count in pair_counts.items() if count == max_collaboration_value]

        end = time.time()
        print(f"Results calcualted in {end-start} seconds")

        return [(max_collaboration_pair, max_collaboration_value) for max_collaboration_pair in max_collaboration_pairs]


def main():
    import sys
    f = open("results.txt", 'w')
    sys.stdout = f

    path = './DBLP/'
    files_name = ['out-dblp_article', 'out-dblp_book', 'out-dblp_incollection', 'out-dblp_inproceedings', 'out-dblp_mastersthesis', 'out-dblp_phdthesis', 'out-dblp_proceedings']
    datasets_name = ['article', 'book', 'incollection', 'inproceedings', 'mastersthesis', 'phdthesis', 'proceedings']
    dblp_list = [DBLP(graph=nx.Graph(), dataset_name=dataset_name) for _, dataset_name in zip(files_name, datasets_name)]
    
    nrows = None
    for i, file_name in enumerate(files_name):
        print(f"Dataset: {file_name}\n") 
        # Read csv file and create pandas Serie
        print("Loading data...\n")
        data = pd.read_csv(f'{path + file_name}.csv', sep=';', dtype=str, nrows=nrows)

        # Remove rows with NaN values in the 'author' field
        data.dropna(subset=['author'], inplace=True)

        dblp_list[i].create_graph_bipartite(data)

        print("\n")
        
        # Exercise 1
        threshold_year = 2020
        max_himself = dblp_list[i].ex_1(threshold_year=threshold_year)
        if max_himself:
            print(f"Author with the most publications written by himself unitl {threshold_year}:")
            [print(f"\t-{author}: {count}") for author, count in max_himself]
        else:
            print(f"\tNo author has written a publication by himself until {threshold_year}!")
        
        print("\n")

        # Exercise 2
        threshold_year = 2020
        print("Finding diameter...")
        diameter = dblp_list[i].ex_2(threshold_year=threshold_year)
        print(f"\tDiameter for graph with publications until {threshold_year}: {diameter}\n")

        # Exercise 3
        threshold_year = 2020
        max_shared_publications = dblp_list[i].ex_3(threshold_year=threshold_year)
        if max_shared_publications:
            print(f"Pair of publications sharing the most authors until {threshold_year}:")
            [print(f"\t-{pair}: {count}") for (pair, count) in max_shared_publications]
        else:
            print(f"\t No pair of articles shares an author until {threshold_year}!")
        
        print("\n--------------------------------------------------\n")

    # Crate union graph renaming nodes
    union_graph: nx.Graph() = nx.union_all([dblp_dataset.graph for dblp_dataset in dblp_list], datasets_name)
    union_dblp = DBLP(graph=union_graph, dataset_name="union")
    print(f"Union graph created! {union_dblp.graph}\n")

    union_dblp.node_id_to_data_id = dict(itertools.chain(
        *map(lambda dblp_dataset, dataset_name: ((f"{dataset_name}{k}", v)
                 for k, v in dblp_dataset.node_id_to_data_id.items()), dblp_list, datasets_name)))

    # Clean graphs
    for dblp_dataset in dblp_list:
        dblp_dataset.graph.clear()
    
    # Exercise 1
    threshold_year = 2020
    max_himself = union_dblp.ex_1(threshold_year=threshold_year)
    if max_himself:
        print(f"Author with the most publications written by himself unitl {threshold_year}:")
        [print(f"\t-{author}: {count}") for author, count in max_himself]
    else:
        print(f"\tNo author has written an publication by himself until {threshold_year}!")

    print("\n")

    # Exercise 2
    threshold_year = 2020
    print("Finding diameter...")
    diameter = union_dblp.ex_2(threshold_year=threshold_year)
    print(f"\tDiameter for graph with publications until {threshold_year}: {diameter}\n")

    # Exercise 3
    threshold_year = 1980
    max_shared_publications = union_dblp.ex_3(threshold_year=threshold_year)
    if max_shared_publications:
        print(f"Pair of publications sharing the most authors until {threshold_year}:")
        [print(f"\t-{pair}: {count}") for (pair, count) in max_shared_publications]
    else:
        print(f"\tNo pair of articles shares an author until {threshold_year}!")
    
    # Exercise 4
    max_collaboration_authors = union_dblp.ex_4()
    if max_collaboration_authors:
        print(f"Pair of authors with most collaboration:")
        [print(f"\t-{pair}: {count}") for (pair, count) in max_collaboration_authors]
    else:
        print(f"\tNo pair of authors collaborates!")
    
    print("\n--------------------------------------------------\n")
    
    f.close()

if __name__ == "__main__":
    main()