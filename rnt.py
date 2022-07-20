"""
Package name: 'rnt' (Reddit Network Toolkit)
Version number: 0.1.1,
Author: Jacob A. Rohde,
Author_email: jarohde1@gmail.com
Description: A simple tool for generating and analyzing Reddit networks.
Github url: https://github.com/jarohde/rnt
License: MIT
"""

__all__ = ['GetRedditData', 'GetRedditNetwork', 'subreddit_statistics', 'reddit_thread_statistics',
           'merge_reddit_submissions_and_comments', 'single_network_plot']

import pandas as pd
import networkx as nx
from pmaw import PushshiftAPI
from datetime import datetime, timedelta
from math import nan
pd.options.mode.chained_assignment = None


class GetRedditData:
    """
    A class object for extracting a Reddit data set.

    Arguments:

    - search_term: The only required parameter; takes a string object (or list of strings) as a keyword for
      searching Reddit submissions.

    - search_term_is_subreddit: Optional Boolean (True or False) argument to signify whether GetRedditData extracts
      a subreddit data set; default set to False.

    - size: Optional integer argument to signify how many Reddit submissions and their associated comments to
      extract; default set to 500 submission. GetRedditData should only be used to extract limited or exploratory
      data sets. I recommend using the Pushshift Reddit repository for extracting large data sets.

    - start_date/end_date: Optional date arguments for GetRedditData; default end date set to current date and
      default start date set to one week prior. Format should be string objects organized like 'YYYY, MM, DD'
      (e.g., start_date='2022, 5, 27' for May 27, 2022).

    Attributes:

    - GetRedditData.df: Extracts the Reddit data set as a pandas DataFrame object.

    Methods:

    - GetRedditData.write_data(): Object method that writes the pandas DataFrame object to file. The method can take
      file_type and file_name as optional arguments. file_type indicates what file format to use when writing the data
      set and accepts a string argument of either 'json' or 'csv'; default set to 'json'. file_name takes a string to
      indicate what the file name should be saved as; default set to the search term provided.
    """

    def __init__(self, search_term, **kwargs):

        self.search_term = search_term
        self.search_term_is_subreddit = kwargs.get('search_term_is_subreddit', False)
        self.start_date = kwargs.get('start_date', None)
        self.end_date = kwargs.get('end_date', None)
        self.size = kwargs.get('size', 500)
        self.df = pd.DataFrame

        if self.search_term_is_subreddit is False:
            if type(self.search_term) == list:
                self.search_term = '|'.join(self.search_term)

        if str(self.size).lower() == 'all':
            limit = None

        else:
            limit = self.size

        today = int(str(datetime.now().timestamp()).split('.')[0])

        if self.start_date is not None:
            # Format: '2022, 5, 27' for May 27, 2022
            after = [int(d) for d in self.start_date.split(',')]
            after = int(datetime(after[0], after[1], after[2], 0, 0).timestamp())

        else:
            after = int(str(datetime.timestamp(datetime.today() - timedelta(days=7))).split('.')[0])

        if self.end_date is not None:
            # Format: '2022, 5, 27' for May 27, 2022
            before = [int(d) for d in self.end_date.split(',')]
            before = int(datetime(before[0], before[1], before[2], 0, 0).timestamp())

        else:
            before = today

        if self.search_term_is_subreddit:
            collection_type = 'subreddit.'

        else:
            collection_type = 'search term(s).'

        if limit is None:
            search_string_prompt = f'Collecting all submissions and comments from the "{self.search_term}" ' \
                                   f'{collection_type}'

        else:
            search_string_prompt = f'Collecting {limit} submissions and their comments from the "{self.search_term}" ' \
                                   f'{collection_type}'

        print(search_string_prompt)

        api = PushshiftAPI()

        submission_kwargs = {'limit': limit, 'after': after, 'before': before, 'mem_safe': True}

        if self.search_term_is_subreddit:
            submission_kwargs['subreddit'] = self.search_term

        else:
            submission_kwargs['q'] = self.search_term

        submissions_data = api.search_submissions(**submission_kwargs)

        submissions_data = [row for row in submissions_data]

        submission_ids = [row['id'] for row in submissions_data if row['num_comments'] > 0]

        comment_kwargs = {'q': '*', 'link_id': submission_ids, 'mem_safe': True}

        comments_data = api.search_comments(**comment_kwargs)

        comments_data = [comment for comment in comments_data]

        collected_data_list = submissions_data + comments_data

        title, body, selftext, author, score, created_utc, id, \
            link_id, parent_id, subreddit, num_comments, distinguished, post_type = \
            [], [], [], [], [], [], [], [], [], [], [], [], []

        for row in collected_data_list:
            author.append(row['author'])
            score.append(row['score'])
            created_utc.append(row['created_utc'])
            id.append(row['id'])
            subreddit.append(row['subreddit'])

            if 'link_id' in row.keys():
                title.append('')
                selftext.append('')
                body.append(row['body'])
                num_comments.append('')
                link_id.append(row['link_id'])
                parent_id.append(row['parent_id'])
                distinguished.append(row['distinguished'])
                post_type.append('c')

            else:
                title.append(row['title'])
                body.append('')
                num_comments.append(row['num_comments'])
                link_id.append('')
                parent_id.append('')
                distinguished.append('')
                post_type.append('s')
                try:
                    selftext.append(row['selftext'])
                except KeyError:
                    selftext.append('')

        df = pd.DataFrame(list(zip(title, body, selftext, author, score, created_utc, id,
                                   link_id, parent_id, subreddit, num_comments, distinguished, post_type)),
                          columns=['title', 'body', 'selftext', 'author', 'score', 'created_utc', 'id',
                                   'link_id', 'parent_id', 'subreddit', 'num_comments', 'distinguished', 'post_type'])

        df['created'] = df.created_utc.apply(lambda x: datetime.utcfromtimestamp(x))

        df = df[['author', 'title', 'selftext', 'body', 'id', 'subreddit', 'num_comments', 'parent_id',
                 'score', 'distinguished', 'link_id', 'created_utc', 'created', 'post_type']]

        df = df.sort_values(by='created_utc')
        df = df.reset_index(drop=True)

        self.df = df

    def __repr__(self):

        message = (f"Reddit data object:\n"
                   f"Search term(s): {self.search_term}\n"
                   f"Search term is subreddit: {self.search_term_is_subreddit}\n"
                   f"Total dataframe size: {len(self.df)}\n"
                   f"Number of Reddit submissions: {len(self.df.loc[self.df.post_type == 's'])}\n"
                   f"Number of Reddit comments: {len(self.df.loc[self.df.post_type == 'c'])}\n")

        return message

    def write_data(self, **kwargs):
        """
        A method for saving Reddit data to file.

        Arguments:

        - file_type: Accepts a string of either 'json' or 'csv'; default set to 'json'.

        - file_name Accepts a string to indicate what the file name should be saved as; default set to the search term
          provided.
        """

        file_type = kwargs.get('file_type', 'json')
        file_name = kwargs.get('file_name', self.search_term)

        if file_type.lower() != 'json' and file_type.lower() != 'csv' or type(file_type) != str:
            return 'Error: File type only supports .csv or .json extensions.'

        else:
            name = f'{file_name}.{file_type.lower()}'
            print(f'Writing {name} to file.')

            if file_type.lower() == 'json':
                self.df.to_json(name, orient='records', lines=True)

            if file_type.lower() == 'csv':
                self.df.to_csv(name, index=False, header=True, encoding='utf-8', na_rep='nan')


class GetRedditNetwork:
    """
    A class object for generating edge and node lists, and a NetworkX graph object from a Reddit data set.

    Arguments:

    - reddit_dataset: The only required argument. Takes a Reddit data set or a GetRedditData object.

    - edge_type: Optional string argument of either 'directed' or 'undirected' to signify network edge type; default
      set to directed.

    - text_attribute: Optional string, list, or dictionary argument to characterize an edge attribute based on one or
      more text categories. Result will return True or False for a network edge if the Reddit submission initiating the
      edge contains the provided keyword(s). Providing the argument with a string or list data type will generate a
      single text attribute column in the edge list and NetworkX graph object. Providing the argument with a dictionary
      data type will generate multiple text attribute columns. Dictionary text attribute example:

      text_attribute={'apples': ['fuji', 'red delicious', 'granny smith'],
                      'oranges': ['valencia', 'mandarin', 'tangerine'],
                      'berries': ['blueberry', 'raspberry', 'blackberry']}

    Attributes:

    - GetRedditNetwork.edge_list: Returns a pandas DataFrame of the network edge list with columns for the poster,
      commenter, the subreddit the edge occurred in, and an optional text attribute column.

    - GetRedditNetwork.node_list: Returns a pandas DataFrame of the network node list with columns for each unique node,
      the node's in-degree and out-degree values, and a list of subreddits the node participated in within the network.

    - GetRedditNetwork.graph: Returns a NetworkX graph object.

    Methods:

    - GetRedditData.write_data(): Object method that writes the edge_list and node_list attributes to file. The method
      can take file_type as an optional parameter. file_type indicates what file format to use when writing the data
      sets and accepts a string argument of either 'json' or 'csv'; default set to 'json'.
    """

    def __init__(self, reddit_dataset, **kwargs):

        self.reddit_dataset = reddit_dataset
        self.edge_type = kwargs.get('edge_type', 'directed')
        self.text_attribute = kwargs.get('text_attribute', False)
        self.edge_by = kwargs.get('edge_by', 'link_id')
        self.edge_list = pd.DataFrame
        self.node_list = pd.DataFrame
        self.graph = nx.Graph

        if 'GetRedditData' in str(type(self.reddit_dataset)):
            df = self.reddit_dataset.df

        else:
            df = self.reddit_dataset

        df = df.loc[df.author != '[deleted]']
        df.fillna('', inplace=True)
        df = add_post_type_column(df)

        if self.text_attribute is not False:
            try:
                if type(self.text_attribute) is list:
                    self.text_attribute = '|'.join(self.text_attribute)

                if type(self.text_attribute) is str:
                    text_attribute = self.text_attribute.lower()

                    df['text_attribute'] = df.title.str.lower().str.contains(text_attribute) | \
                                           df.selftext.str.lower().str.contains(text_attribute) | \
                                           df.body.str.lower().str.contains(text_attribute)

                    df.text_attribute = df.text_attribute.apply(lambda x: str(x))

                if type(self.text_attribute) is dict:
                    for key in self.text_attribute:
                        if type(self.text_attribute[key]) is list:
                            values = '|'.join(self.text_attribute[key])
                        else:
                            values = self.text_attribute[key]

                        df[f'text_attribute_{key}'] = df.title.str.lower().str.contains(values) | \
                                  df.selftext.str.lower().str.contains(values) | \
                                  df.body.str.lower().str.contains(values)

                        df[f'text_attribute_{key}'] = df[f'text_attribute_{key}'].apply(lambda x: str(x))

                text_attribute_columns = [col for col in df.columns if 'text_attribute' in col]

            except (TypeError, AttributeError) as e:
                print(f'Incorrect text attribute input (returned {e} error); ignoring.')
                self.text_attribute = False

        nodes = df.author.unique()
        threads = df.loc[df.post_type == 's'].id.unique()

        if self.edge_by.lower() == 'parent_id':
            threads = df.id.unique()

        edge_dfs = []

        for thread in threads:
            thread_df = df.loc[df[self.edge_by].apply(lambda x: str(x)[3:] == thread)]

            author = list(df.loc[df.id == thread].author)[0]
            thread_subreddit = list(df.loc[df.id == thread].subreddit)[0]

            commenters = [commenter for commenter in thread_df.author if commenter != author]

            commenter_list = []

            if len(commenters) == 0:
                commenter_list.append('')

            else:
                commenter_list = commenter_list + commenters

            author_list = [author] * len(commenter_list)
            subreddit_list = [thread_subreddit] * len(commenter_list)

            thread_edge_df = pd.DataFrame(list(zip(author_list, commenter_list, subreddit_list)),
                             columns=['poster', 'commenter', 'subreddit'])

            if self.text_attribute is not False:
                for col in text_attribute_columns:
                    thread_edge_df[col] = list(df.loc[df.id == thread][col])[0]

            edge_dfs.append(thread_edge_df)

        edge_list = pd.concat(edge_dfs, ignore_index=True)
        edge_list = edge_list.loc[edge_list.commenter != '']

        edge_list = edge_list.reset_index(drop=True)

        in_degree_values, out_degree_values, node_subreddits = [], [], []

        for node in nodes:
            in_degree_values.append(len(edge_list.loc[edge_list.poster == node]))
            out_degree_values.append(len(edge_list.loc[edge_list.commenter == node]))
            node_subreddits.append(list(df.loc[df.author == node].subreddit.unique()))

        node_list = pd.DataFrame(list(zip(nodes, in_degree_values, out_degree_values, node_subreddits)),
                                 columns=['nodes', 'in_degree', 'out_degree', 'node_subreddits'])

        graph_object = nx.DiGraph()

        for node, attribute in zip(nodes, node_subreddits):
            graph_object.add_node(node, subreddit_list=attribute)

        for n1, n2 in zip(edge_list.poster, edge_list.commenter):
            graph_object.add_edge(n1, n2)

        attrs = {}
        edge_list_index = 0
        for n1, n2, subreddit in zip(edge_list.poster, edge_list.commenter, edge_list.subreddit):
            if (n1, n2) not in attrs.keys():
                edge_attr_dict = {}
                edge_attr_dict = {**edge_attr_dict, **{'subreddit': subreddit,
                                                       'weight': len(edge_list[(edge_list.poster == n1) &
                                                                               (edge_list.commenter == n2)])}}

                if self.text_attribute is not False:
                    for col in text_attribute_columns:
                        edge_attr_dict = {**edge_attr_dict, **{col: edge_list[col][edge_list_index]}}

                edge_list_index += 1
                attrs[(n1, n2)] = edge_attr_dict

            else:
                continue

        nx.set_edge_attributes(graph_object, attrs)

        if self.edge_type.lower() == 'undirected':
            graph_object = graph_object.to_undirected()

        self.edge_list = edge_list
        self.node_list = node_list
        self.graph = graph_object

    def __repr__(self):

        message = (f"Reddit network object:\n"
                   f"Number of nodes: {len(self.node_list)}\n"
                   f"Number of edges: {len(self.edge_list)}")

        return message

    def write_data(self, **kwargs):
        """
        A method for saving the edge and node data to file.

        Parameter:

        - file_type: Accepts a string of either 'json' or 'csv'; default set to 'json'.

        - file_name: Accepts a string to indicate what the edge and node list files should be saved as; default set to
          'edge_list' and 'node_list'.
        """

        file_type = kwargs.get('file_type', 'json')
        file_name = kwargs.get('file_name', '')
        file_list = ['node_list', 'edge_list']

        if file_type.lower() != 'json' and file_type.lower() != 'csv' or type(file_type) != str:
            return 'Error: File type only supports .csv or .json extensions.'

        for file in file_list:
            if type(file_name) is str and file_name != '':
                f_name = f'{file}_{file_name}'
            else:
                f_name = file

            name = f'{f_name}.{file_type.lower()}'
            print(f'Writing {f_name} to file.')

            if file_type.lower() == 'json':
                if file == 'node_list':
                    self.node_list.to_json(name, orient='records', lines=True)
                else:
                    self.edge_list.to_json(name, orient='records', lines=True)

            if file_type.lower() == 'csv':
                if file == 'node_list':
                    self.node_list.to_csv(name, index=False, header=True, encoding='utf-8', na_rep='nan')
                else:
                    self.edge_list.to_csv(name, index=False, header=True, encoding='utf-8', na_rep='nan')


def subreddit_statistics(reddit_dataset, subreddit_list=None):
    """
    A function for extracting basic statistics for single or batch subreddit networks.

    Arguments:

    - reddit_dataset: The only required argument. Takes a Reddit data set or a GetRedditData object.

    - subreddit_list: An optional list argument to indicate the specific subreddits to compute analyses for; default
      set to all unique subreddits in a data set that Reddit submissions were published in.

    Returns: The function currently returns a single pandas DataFrame with example subreddit network statistics
    including number of nodes, edges, and network density, among others.
    """

    if 'GetRedditData' in str(type(reddit_dataset)):
        reddit_dataset = reddit_dataset.df

    df = reddit_dataset.loc[reddit_dataset.author != '[deleted]']
    df.fillna('', inplace=True)

    df = add_post_type_column(df)

    if subreddit_list is None:
        subreddit_list = list(df.loc[df.post_type == 's'].subreddit.unique())

    if subreddit_list is not None:
        if type(subreddit_list) is list:
            subreddit_list = subreddit_list

        else:
            print('Incorrect subreddit_list argument; computing data for all subreddits in provided data.')
            subreddit_list = list(df.loc[df.post_type == 's'].subreddit.unique())

    first_subreddit = True
    for subreddit in subreddit_list:
        subreddit_df = df.loc[df.subreddit == subreddit]
        subreddit_graph = GetRedditNetwork(subreddit_df).graph

        connected_components = sorted(nx.strongly_connected_components(subreddit_graph), key=len, reverse=True)
        strongest_connected_component = subreddit_graph.subgraph(connected_components[0])
        degrees = [subreddit_graph.degree(node) for node in subreddit_graph.nodes()]

        subreddit_stats_dict = {'subreddit': subreddit,
                                'num_overall_posts': len(subreddit_df),
                                'num_submissions': len(subreddit_df.loc[subreddit_df.post_type == 's']),
                                'num_comments': len(subreddit_df.loc[subreddit_df.post_type == 'c']),
                                'num_unique_users': len(subreddit_df.loc[subreddit_df.author != '[deleted]'].author.unique()),
                                'density': nx.density(subreddit_graph),
                                'number_graph_edges': len(subreddit_graph.edges()),
                                'number_graph_nodes': len(subreddit_graph.nodes()),
                                'strongest_connected_component_size': len(strongest_connected_component.nodes),
                                'mean_graph_degree': pd.Series(degrees).mean(),
                                'median_graph_degree': pd.Series(degrees).median()}

        if first_subreddit:
            subreddit_batch_statistics_df = pd.DataFrame(subreddit_stats_dict, index=range(0, 1))
            first_subreddit = False

        else:
            subreddit_row = pd.DataFrame(subreddit_stats_dict, index=range(0, 1))
            subreddit_batch_statistics_df = pd.concat([subreddit_batch_statistics_df, subreddit_row])

    return subreddit_batch_statistics_df.reset_index(drop=True)


def reddit_thread_statistics(reddit_dataset, reddit_thread_list=None):
    """
    A function for extracting basic statistics for single or batch Reddit threads (initiated by Reddit
    submissions).

    Arguments:

    - reddit_dataset: The only required argument. Takes a Reddit data set or a GetRedditData object.

    - reddit_thread_list: An optional list argument to provide the specific Reddit thread IDs (i.e., Reddit submission
      IDs) to analyze; default set to all unique threads in a Reddit data set.

    Returns: The function currently returns a single pandas DataFrame with example subreddit network statistics
    including number of nodes, edges, and network density, among others.
    """

    if 'GetRedditData' in str(type(reddit_dataset)):
        reddit_dataset = reddit_dataset.df

    df = reddit_dataset.loc[reddit_dataset.author != '[deleted]']
    df.fillna('', inplace=True)

    df = add_post_type_column(df)

    if reddit_thread_list is not None:
        if type(reddit_thread_list) is list:
            reddit_thread_list = reddit_thread_list

        else:
            print('Incorrect reddit_thread_list argument; computing data for all reddit threads in provided data.')
            reddit_thread_list = list(df.loc[df.post_type == 's'].id.unique())
    else:
        reddit_thread_list = list(df.loc[df.post_type == 's'].id.unique())

    first_thread = True
    for thread in reddit_thread_list:
        thread_df = df.loc[(df.link_id.apply(lambda x: x[3:] == thread)) |
                           (df.id == thread)]

        author_df = thread_df.loc[thread_df.post_type == 's']
        thread_author = list(thread_df.loc[thread_df.post_type == 's'].author)[0]
        date_format_str = '%Y-%m-%d %H:%M:%S'
        author_date = str(list(author_df.created_utc.apply(lambda x: datetime.utcfromtimestamp(x)))[0])
        author_date_formatted = datetime.strptime(author_date, date_format_str)

        commenter_df = thread_df.loc[(thread_df.author != thread_author) &
                                     (thread_df.post_type == 'c')]

        num_responses = len(commenter_df)

        if num_responses > 0:
            num_unique_responders = len(commenter_df.author.unique())
            commenter_response_times = []

            for index, post in commenter_df.iterrows():
                posted_time = str(datetime.utcfromtimestamp(post.created_utc))
                posted_time_formatted = datetime.strptime(posted_time, date_format_str)
                time_differential = (posted_time_formatted - author_date_formatted).total_seconds()
                commenter_response_times.append(time_differential)

            earliest_response_time = pd.Series(commenter_response_times).min()
            latest_response_time = pd.Series(commenter_response_times).max()
            mean_response_time = pd.Series(commenter_response_times).mean()
            std_dev_response_time = pd.Series(commenter_response_times).std()
            median_response_time = pd.Series(commenter_response_times).median()

        else:
            num_unique_responders = 0
            earliest_response_time = nan
            latest_response_time = nan
            mean_response_time = nan
            std_dev_response_time = nan
            median_response_time = nan

        thread_dict = {'author': thread_author,
                       'thread_id': thread,
                       'date': list(author_df.created)[0],
                       'subreddit': list(author_df.subreddit)[0],
                       'distinguished': list(author_df.distinguished)[0],
                       'score': list(author_df.score)[0],
                       'num_unique_responders': num_unique_responders,
                       'earliest_response_time': earliest_response_time,
                       'latest_response_time': latest_response_time,
                       'mean_response_time': mean_response_time,
                       'std_dev_response_time_seconds': std_dev_response_time,
                       'median_response_time': median_response_time}

        if first_thread:
            thread_batch_dict = pd.DataFrame(thread_dict, index=range(0, 1))
            first_thread = False

        else:
            thread_row = pd.DataFrame(thread_dict, index=range(0, 1))
            thread_batch_dict = pd.concat([thread_batch_dict, thread_row])

        thread_statistics_df = thread_batch_dict.reset_index(drop=True)

    return thread_statistics_df


def merge_reddit_submissions_and_comments(submissions_dataset, comments_dataset):
    """
    A function for merging separate Reddit submission and comment data sets into a single data set
    (currently functional but a work in progress).
    """

    req_subs_columns = ['author', 'created_utc', 'subreddit', 'id']
    req_coms_columns = ['author', 'created_utc', 'subreddit', 'link_id']

    submissions_dataset['post_type'] = 's'
    comments_dataset['post_type'] = 'c'

    subs_check = any(col_name not in submissions_dataset.columns for col_name in req_subs_columns)
    coms_check = any(col_name not in comments_dataset.columns for col_name in req_coms_columns)

    if subs_check or coms_check:
        if subs_check and coms_check:
            string_return = 'the submissions and comments data sets'
        elif subs_check:
            string_return = 'the submissions data set'
        else:
            string_return = 'the comments data set'

        return print(f'Incorrect columns in {string_return}.')

    optional_subs_columns = [col for col in submissions_dataset.columns if col not in req_subs_columns]
    optional_coms_columns = [col for col in comments_dataset.columns if col not in req_coms_columns]

    merged_columns = set(optional_subs_columns +
                         optional_coms_columns +
                         req_subs_columns +
                         req_coms_columns)

    for column in merged_columns:
        if column not in submissions_dataset.columns:
            submissions_dataset[column] = ''
        if column not in comments_dataset.columns:
            comments_dataset[column] = ''

    df = pd.concat([submissions_dataset, comments_dataset])

    df['created'] = df.created_utc.apply(lambda x: datetime.utcfromtimestamp(int(x)))

    df = df[['author', 'title', 'selftext', 'body', 'id', 'subreddit', 'num_comments', 'parent_id',
             'score', 'distinguished', 'link_id', 'created_utc', 'created', 'post_type']]

    df = df.sort_values(by='created_utc')
    df = df.reset_index(drop=True)

    return df


def add_post_type_column(df):
    """
    An internal function for identifying Reddit dataset columns and appending a post_type column.
    """

    if 'post_type' not in df.columns:
        post_type = []
        for row in df.link_id:
            if row == '':
                post_type.append('s')
            else:
                post_type.append('c')
        df['post_type'] = post_type

    return df


def single_network_plot(network=None, **kwargs):

    """
    A simple function for plotting networks via NetworkX and Matplotlib (additional install required). Please note this
    function is currently a work in progress and is meant to be basic tool to plot a single graph. See NetworkX
    documentation for more advanced plotting needs.

    Arguments:

    - network: The only required argument. Takes a GetRedditNetwork or NetworkX graph object.

    - title: Optional string argument to add a title to the plot.

    - pos: Optional string argument to set the NetworkX plotting algorithm. For ease of use, the argument currently
      accepts one of the following layout types as a string:

      - 'spring_layout' (default)
      - 'kamada_kawai_layout'
      - 'circular_layout'
      - 'random_layout'

    - **kwargs: The function also accepts several other NetworkX keyword arguments for plotting (please see NetworkX
      documentation for more info on these arguments). Currently accepted arguments include:

      - 'arrows' (bool)
      - 'arrowsize' (int)
      - 'edge_color' (str or list/array)
      - 'node_color' (str or list/array)
      - 'node_size' (str or list/array)
      - 'width' (int/float or list/array)
      - 'with_labels' (bool)
    """
    import matplotlib.pyplot as plt

    if network is not None:
        if 'GetRedditNetwork' in str(type(network)):
            G = network.graph
        else:
            G = network

    # node attributes
    with_labels = kwargs.get('with_labels', False)
    node_color = kwargs.get('node_color', 'black')
    node_size = kwargs.get('node_size', 30)

    # edge attributes
    edge_color = kwargs.get('edge_color', 'grey')
    width = kwargs.get('width', 1)
    arrows = kwargs.get('arrows', None)
    arrowsize = kwargs.get('arrowsize', 10)

    # general plot attributes
    title = kwargs.get('title', None)
    pos = kwargs.get('pos', 'spring_layout')

    plt.figure()
    plt.title(title)

    layouts = {'spring_layout': nx.spring_layout(G),
               'kamada_kawai_layout': nx.kamada_kawai_layout(G),
               'circular_layout': nx.circular_layout(G),
               'random_layout': nx.random_layout(G)}

    pos = layouts.get(pos.lower(), 'spring_layout')

    plt_kwargs = {'pos': pos,
                  'edge_color': edge_color,
                  'node_color': node_color,
                  'with_labels': with_labels,
                  'node_size': node_size,
                  'width': width,
                  'arrows': arrows,
                  'arrowsize': arrowsize}

    nx.draw_networkx(G, **plt_kwargs)

    plt.show()
