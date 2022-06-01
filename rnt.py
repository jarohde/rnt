import pandas as pd
from pmaw import PushshiftAPI
from datetime import datetime, timedelta
import networkx as nx
from math import nan
pd.options.mode.chained_assignment = None


class GetRedditData:

    def __init__(self, search_term, search_term_is_subreddit=False, size=None, start_date=None, end_date=None):
        self.search_term = search_term
        self.size = size
        self.start_date = start_date
        self.end_date = end_date
        self.search_term_is_subreddit = search_term_is_subreddit
        self.df = None

        if search_term_is_subreddit is False:
            if type(self.search_term) == list:
                self.search_term = '|'.join(self.search_term)

        if size is None:
            self.size = 500

        elif type(size) is str and size.lower() == 'all':
            self.size = None

        else:
            self.size = size

        today = int(str(datetime.now().timestamp()).split('.')[0])

        if start_date is not None:
            # Format: '2022, 5, 27' for May 27, 2022
            start_date = [int(d) for d in start_date.split(',')]
            self.start_date = int(datetime(start_date[0], start_date[1], start_date[2], 0, 0).timestamp())

        else:
            self.start_date = int(str(datetime.timestamp(datetime.today() - timedelta(days=7))).split('.')[0])

        if end_date is not None:
            # Format: '2022, 5, 27' for May 27, 2022
            end_date = [int(d) for d in end_date.split(',')]
            self.end_date = int(datetime(end_date[0], end_date[1], end_date[2], 0, 0).timestamp())

        else:
            self.end_date = today

        if self.search_term_is_subreddit:
            collection_type = 'subreddit.'

        else:
            collection_type = 'search term(s).'

        search_string_prompt = f'Collecting {self.size} submissions and {self.size} comments from the "{self.search_term}" {collection_type}'
        print(search_string_prompt)

        api = PushshiftAPI()

        if search_term_is_subreddit:
            data_submissions = api.search_submissions(subreddit=self.search_term,
                                                      limit=self.size,
                                                      after=self.start_date,
                                                      before=self.end_date,
                                                      mem_safe=True)

            data_comments = api.search_comments(subreddit=self.search_term,
                                                limit=self.size,
                                                after=self.start_date,
                                                before=self.end_date,
                                                mem_safe=True)

        else:
            data_submissions = api.search_submissions(q=self.search_term,
                                                      limit=self.size,
                                                      after=self.start_date,
                                                      before=self.end_date,
                                                      mem_safe=True)

            data_comments = api.search_comments(q=self.search_term,
                                                limit=self.size,
                                                after=self.start_date,
                                                before=self.end_date,
                                                mem_safe=True)

        collected_data_list = [row for row in data_submissions] + [row for row in data_comments]

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

        self.df = df

    def __str__(self):

        message = (f"Reddit data object:\n"
                   f"Search term(s): {self.search_term}\n"
                   f"Search term is subreddit: {self.search_term_is_subreddit}\n"
                   f"Dataframe size: {len(self.df)}\n"
                   f"Collection start date: {str(datetime.utcfromtimestamp(self.start_date)).split()[0]}\n"
                   f"Collection end date: {str(datetime.utcfromtimestamp(self.end_date)).split()[0]}")

        return message


class GetRedditNetwork:

    def __init__(self, reddit_dataset, edge_type='directed', edge_by=None, text_attribute=False):

        if 'GetRedditData' in str(type(reddit_dataset)):
            reddit_dataset = reddit_dataset.df

        self.df = reddit_dataset.loc[reddit_dataset.author != '[deleted]']
        self.edge_by = edge_by
        self.edge_type = edge_type
        self.text_attribute = text_attribute
        self.edge_list = None
        self.node_list = None
        self.graph = None

        self.df.fillna('', inplace=True)

        if 'post_type' not in self.df.columns:
            post_type = []
            for row in self.df.link_id:
                if row == '':
                    post_type.append('s')
                else:
                    post_type.append('c')
            self.df['post_type'] = post_type

        if self.text_attribute is not False:
            try:
                if type(self.text_attribute) is list:
                    self.text_attribute = '|'.join(self.text_attribute)
                self.text_attribute = self.text_attribute.lower()
                self.df['text_attribute'] = self.df.title.str.lower().str.contains(self.text_attribute) | \
                                            self.df.selftext.str.lower().str.contains(self.text_attribute) | \
                                            self.df.body.str.lower().str.contains(self.text_attribute)

                self.df.text_attribute = self.df.text_attribute.apply(lambda x: str(x))

            except (TypeError, AttributeError) as e:
                print(f'Incorrect text attribute input (returned {e} error); ignoring.')
                self.text_attribute = False

        nodes = self.df.author.unique()
        threads = self.df.loc[self.df.post_type == 's'].id.unique()
        thread_separator = 'link_id'

        if self.edge_by is not None and self.edge_by.lower() == 'parent':
            threads = self.df.id.unique()
            thread_separator = 'parent_id'

        edge_dfs = []

        for thread in threads:
            thread_df = self.df.loc[self.df[thread_separator].apply(lambda x: str(x)[3:] == thread)]

            author = list(self.df.loc[self.df.id == thread].author)[0]
            thread_subreddit = list(self.df.loc[self.df.id == thread].subreddit)[0]

            if self.text_attribute is not False:
                thread_df_attribute = list(self.df.loc[self.df.id == thread].text_attribute)[0]

            commenters = [commenter for commenter in thread_df.author if commenter != author]

            commenter_list = []

            if len(commenters) == 0:
                commenter_list.append('')

            else:
                commenter_list = commenter_list + commenters

            author_list = [author]*len(commenter_list)
            subreddit_list = [thread_subreddit]*len(commenter_list)

            if self.text_attribute is not False:
                text_attribute_list = [thread_df_attribute]*len(commenter_list)

            else:
                text_attribute_list = ['']*len(commenter_list)


            edge_dfs.append(
                pd.DataFrame(list(zip(author_list, commenter_list, subreddit_list, text_attribute_list)),
                             columns=['poster', 'commenter', 'subreddit', 'text_attribute']))

        edge_list = pd.concat(edge_dfs, ignore_index=True)
        edge_list = edge_list.loc[edge_list.commenter != '']

        edge_list = edge_list.reset_index(drop=True)

        in_degree_values, out_degree_values, node_subreddits = [], [], []

        for node in nodes:
            in_degree_values.append(len(edge_list.loc[edge_list.poster == node]))
            out_degree_values.append(len(edge_list.loc[edge_list.commenter == node]))
            node_subreddits.append(list(self.df.loc[self.df.author == node].subreddit.unique()))

        node_list = pd.DataFrame(list(zip(nodes, in_degree_values, out_degree_values, node_subreddits)),
                                 columns=['nodes', 'in_degree', 'out_degree', 'node_subreddits'])

        graph_object = nx.DiGraph()

        for node, attribute in zip(nodes, node_subreddits):
            graph_object.add_node(node, subreddit_list=attribute)

        for n1, n2, sub_attribute, text_attribute in \
                zip(edge_list.commenter, edge_list.poster, edge_list.subreddit, edge_list.text_attribute):
            graph_object.add_edge(n1, n2, subreddit=sub_attribute, text_attribute=text_attribute)

        if self.text_attribute is False:
            del edge_list['text_attribute']

        if self.edge_type.lower() == 'undirected':
            graph_object = graph_object.to_undirected()

        self.edge_list = edge_list
        self.node_list = node_list
        self.graph = graph_object

    def __str__(self):

        message = (f"Reddit network object:\n"
                   f"Number of nodes: {len(self.node_list)}\n"
                   f"Number of edges: {len(self.edge_list)}")

        return message


def subreddit_statistics(reddit_dataset, subreddit_list=None):

    if 'GetRedditData' in str(type(reddit_dataset)):
        reddit_dataset = reddit_dataset.df

    df = reddit_dataset.loc[reddit_dataset.author != '[deleted]']
    df.fillna('', inplace=True)

    if 'post_type' not in df.columns:
        post_type = []
        for row in df.link_id:
            if row == '':
                post_type.append('s')
            else:
                post_type.append('c')
        df['post_type'] = post_type

    if subreddit_list is None:
        subreddit_list = list(df.loc[df.post_type == 's'].subreddit.unique())

    if subreddit_list is not None:
        if type(subreddit_list) is list:
            subreddit_list = subreddit_list

        else:
            print('Incorrect subreddit_list argument; computing data for all subreddits in provided data.')
            subreddit_list = list(df.loc[df.post_type == 's'].subreddit.unique())

    index = 0
    for subreddit in subreddit_list:
        subreddit_df = df.loc[df.subreddit == subreddit]
        subreddit_graph = GetRedditNetwork(subreddit_df, edge_by='link').graph

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

        if index == 0:
            subreddit_batch_statistics_df = pd.DataFrame(subreddit_stats_dict, index=range(0, 1))
            index = 1

        else:
            subreddit_row = pd.DataFrame(subreddit_stats_dict, index=range(0, 1))
            subreddit_batch_statistics_df = pd.concat([subreddit_batch_statistics_df, subreddit_row])

    return subreddit_batch_statistics_df.reset_index(drop=True)


def reddit_thread_statistics(reddit_dataset, reddit_thread_list=None):

    if 'GetRedditData' in str(type(reddit_dataset)):
        reddit_dataset = reddit_dataset.df

    df = reddit_dataset.loc[reddit_dataset.author != '[deleted]']
    df.fillna('', inplace=True)

    if 'post_type' not in df.columns:
        post_type = []
        for row in df.link_id:
            if row == '':
                post_type.append('s')
            else:
                post_type.append('c')
        df['post_type'] = post_type

    if reddit_thread_list is not None:
        if type(reddit_thread_list) is list:
            reddit_thread_list = reddit_thread_list

        else:
            print('Incorrect reddit_thread_list argument; computing data for all reddit threads in provided data.')
            reddit_thread_list = list(df.loc[df.post_type == 's'].id.unique())
    else:
        reddit_thread_list = list(df.loc[df.post_type == 's'].id.unique())

    index = 0
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

        if index == 0:
            thread_batch_dict = pd.DataFrame(thread_dict, index=range(0, 1))
            index = 1

        else:
            thread_row = pd.DataFrame(thread_dict, index=range(0, 1))
            thread_batch_dict = pd.concat([thread_batch_dict, thread_row])

        thread_statistics_df = thread_batch_dict.reset_index(drop=True)

    return thread_statistics_df
