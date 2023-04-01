import re
import itertools
from urllib.parse import urlparse
import pandas as pd

__all__ = ['url_extractor', 'domain_extractor', 'url_reddit_dataset']


def url_extractor(text):
    """
    A function for extracting all URLs from a single text document. The only required argument is a string.
    """

    url_expression = '(http|ftp|https)(:\/\/)([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'

    if type(text) != str:
        print('ERROR: Incorrect data type! url_extractor requires a single string as an argument.')
        return

    else:
        text = text.lower()
        if len(re.findall(url_expression, text)) > 0:
            url_list = [l[0] + l[1] + l[2] + l[3] for l in re.findall(url_expression, text)]
            return url_list
        else:
            return []


def domain_extractor(text):
    """
    A function for extracting URL domains from a string containing hyperlinks.
    """

    link_list = url_extractor(text)
    domain_list = [re.sub('www\.', '', urlparse(domain).netloc) for domain in link_list]
    return domain_list


def url_reddit_dataset(reddit_dataset, **kwargs):
    """
    A function to generate a "URL data set" where each row in the data set is a unique URL.
    """

    url, domain, frequency, subreddits, number_of_subreddits, authors, number_of_authors, \
        post_types = [], [], [], [], [], [], [], []

    domain_format = kwargs.get('domain_format', False)

    if type(domain_format) is not bool:
        domain_format = False

    if 'url_list' not in reddit_dataset.columns:
        print('ERROR: Data set missing a column title "url_list" containing text-related URLs.')
        return

    else:
        if domain_format:
            links = itertools.chain(*list(reddit_dataset.domain_list))
        else:
            links = itertools.chain(*list(reddit_dataset.url_list))

        links = set([link for link in links])
        index = 1
        for link in links:
            if domain_format:
                url.append('')
                domain.append(link)
                subset_df = reddit_dataset.loc[reddit_dataset.domain_list.apply(lambda x: link in x)]

            else:
                url.append(link)
                domain.append(re.sub('www\.', '', urlparse(link).netloc))
                subset_df = reddit_dataset.loc[reddit_dataset.url_list.apply(lambda x: link in x)]

            frequency.append(len(subset_df))

            # Extracting post type information
            if 'post_type' in reddit_dataset.columns:
                all_post_types = list(subset_df.post_type)
                post_types.append({i: all_post_types.count(i) for i in set(all_post_types)})
            else:
                post_types.append('')

            # Extracting subreddit information
            all_subreddits = list(subset_df.subreddit)
            subreddits.append({i: all_subreddits.count(i) for i in set(all_subreddits)})
            number_of_subreddits.append(len({i: all_subreddits.count(i) for i in set(all_subreddits)}))

            # Extracting author information
            all_authors = list(subset_df.author)
            authors.append({i: all_authors.count(i) for i in set(all_authors)})
            number_of_authors.append(len({i: all_authors.count(i) for i in set(all_authors)}))

            if len(links) > 1000 and (len(links) - index) % 100 == 0:
                print(f'{len(links) - index} URLs left.')

            index += 1

        url_df = pd.DataFrame(list(zip(url, domain, frequency, subreddits, number_of_subreddits,
                                       authors, number_of_authors, post_types)),
                              columns=['url', 'domain', 'frequency', 'subreddits', 'number_of_subreddits',
                                       'authors', 'number_of_authors', 'post_types'])

        if domain_format:
            url_df.drop('url', axis=1, inplace=True)

        if 'post_type' not in reddit_dataset.columns:
            url_df.drop('post_types', axis=1, inplace=True)

        url_df = url_df.reset_index(drop=True)

        return url_df
