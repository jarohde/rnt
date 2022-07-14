# RNT - Reddit Network Toolkit

A simple tool for generating and extracting network objects from Reddit data sets.

## Author

**Jacob Rohde**

Email: [jarohde1\@gmail.com](mailto:jarohde1@gmail.com) \| Twitter: [\@jacobrohde](https://twitter.com/JacobRohde) \| GitHub: [\@jarohde](https://github.com/jarohde)

## Features

-   Extracts a simple data set of Reddit submissions and their associated comments via keyword or subreddit search terms.

-   Provides single and batch subreddit- and thread-level network statistics.

-   Generates edge and node lists, and creates network objects (via NetworkX) from Reddit data sets. Networks:

    -   can be directed or undirected;
    -   contain subreddit node and edge attributes;
    -   allow for optional text classification attributes.

## General

**Current version:**

0.1.0 (released 07/14/22)

**Import RNT library:**

    import rnt

**Classes and functions**

-   `GetRedditData()`

-   `GetRedditNetwork()`

-   `subreddit_statistics()`

-   `reddit_thread_statistics()`

## Usage

### GetRedditData()

    rnt.GetRedditData(search_term, 
                      search_term_is_subreddit, 
                      size, 
                      start_date, 
                      end_date)

**Overview:** A class object for extracting a Reddit data set.

**Arguments/attributes:**

`search_term`: The only required argument. Takes a string as a single search term or list of strings for multiple search terms (e.g., `search_term='news'` or `search_term=['news', 'cnn']`). If extracting a subreddit data set (see '`search_term_is_subreddit`' below), only provide a string of a single subreddit name (e.g., 'AskReddit').

`search_term_is_subreddit`: Optional Boolean (True or False) argument to signify whether `GetRedditData` extracts a subreddit data set; default set to False.

`size`: Optional integer argument to signify how many Reddit submissions and their associated comments to extract; default set to 500 submission. `GetRedditData` should only be used to extract limited or exploratory data sets. I recommend using the Pushshift Reddit repository for extracting large data sets.

`start_date`/`end_date`: Optional date arguments for `GetRedditData`; default end date set to current date and default start date set to one week prior. Format should be string objects organized like 'YYYY, MM, DD' (e.g., `start_date='2022, 5, 27'` for May 27, 2022).

`GetRedditData.df`: Object attribute; extracts the Reddit data set as a pandas DataFrame object.

`GetRedditData.write_data()`: Object method that writes the pandas DataFrame object to file. The method can take `file_type` and `file_name` as optional parameters. `file_type` indicates what file format to use when writing the data set and accepts a string argument of either 'json' or 'csv'; default set to 'json'. `file_name` takes a string to indicate what the file name should be saved as; default set to the search term provided.

### GetRedditNetwork()

    rnt.GetRedditNetwork(reddit_dataset, 
                         edge_type, 
                         text_attribute) 

**Overview:** A class object for generating edge and node lists, and a NetworkX graph object from a Reddit data set.

**Arguments/attributes:**

`reddit_dataset`: The only required argument. Takes a Reddit data set or a `GetRedditData` object.

`edge_type`: Optional string argument of either 'directed' or 'undirected' to signify network edge type; default set to directed.

`text_attribute`: Optional string, list, or dictionary argument to characterize an edge attribute based on one or more text categories. Result will return True or False for a network edge if the Reddit submission initiating the edge contains the provided keyword(s). Providing the argument with a string or list data type will generate a single text attribute column in the edge list and NetworkX graph object. Providing the argument with a dictionary data type will generate multiple text attribute columns. Dictionary text attribute example:

    text_attribute={'apples': ['fuji', 'red delicious', 'granny smith'], 
                    'oranges': ['valencia', 'mandarin', 'tangerine'], 
                    'berries': ['blueberry', 'raspberry', 'blackberry']}

`GetRedditNetwork.edge_list`: Returns a pandas DataFrame of the network edge list with columns for the poster, commenter, the subreddit the edge occurred in, and an optional text attribute column.

`GetRedditNetwork.node_list`: Returns a pandas DataFrame of the network node list with columns for each unique node, the node's in-degree and out-degree values, and a list of subreddits the node participated in within the network.

`GetRedditNetwork.graph`: Returns a NetworkX graph object.

`GetRedditNetwork.write_data()`: Object method that writes the `edge_list` and `node_list` attributes to file. The method can take `file_type` and `file_name` as optional parameters. `file_type` indicates what file format to use when writing the data sets and accepts a string argument of either 'json' or 'csv'; default set to 'json'. `file_name` takes a string to indicate what to append at the end of the edge and node list file names (e.g., `file_name='apple'` will save the files as 'edge_list_apple.json' and 'node_list_apple.json').

### subreddit_statistics()

    rnt.subreddit_statistics(reddit_dataset, subreddit_list) 

**Overview:** A function for extracting basic statistics for single or batch subreddit networks. The function currently returns a single pandas DataFrame with example subreddit network statistics including number of nodes, edges, and network density, among others.

`reddit_dataset`: The only required argument. Takes a Reddit data set or a `GetRedditData` object.

`subreddit_list`: An optional list argument to indicate the specific subreddits to compute analyses for; default set to all unique subreddits in a data set that Reddit submissions were published in.

### reddit_thread_statistics()

    rnt.reddit_thread_statistics(reddit_dataset, reddit_thread_list)

**Overview:** A function for extracting basic statistics for single or batch Reddit threads (initiated by Reddit submissions). The function currently returns a single pandas DataFrame with example statistics including the number of unique commenters to the thread, and the earliest/latest response times to the thread, among others.

`reddit_dataset`: The only required argument. Takes a Reddit data set or a `GetRedditData` object.

`reddit_thread_list`: An optional list argument to provide the specific Reddit thread IDs (i.e., Reddit submission IDs) to analyze; default set to all unique threads in a Reddit data set.

## Requirements

-   Python 3.XX
-   pandas - a Python library for data management.
-   NetworkX - a Python library for network analysis.
-   PMAW - a multithread tool for extracting Reddit data via the [Pushshift API](https://pushshift.io/api-parameters/)

## Support

For support, email [jarohde1\@gmail.com](mailto:jarohde1@gmail.com).

## License

[MIT](https://choosealicense.com/licenses/mit/)
