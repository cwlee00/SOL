import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def tsne_visualize(features, labels, n_components=2, perplexity=40, n_iter=300, verbose=1, return_tsne=False, show_legend=True):
    """
    :param features: input features
    :param labels: feature-wise label
    :param n_components: dimension of embedded space
    :return: TSNE visualization
    """
    fdim = features.shape[-1]
    if fdim > 2:
        tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
        tsne_results = tsne.fit_transform(features)
    else:
        tsne_results = features

    classes = np.unique(labels)

    fig, ax = plt.subplots(figsize=(20, 20))
    for color_idx, cls in enumerate(classes):
        idx = np.argwhere(labels == cls)
        idx = np.squeeze(idx)
        cur_cls_feats = tsne_results[idx, ...]
        cur_cls_feats = np.reshape(cur_cls_feats, [-1, 2])
        ax.scatter(cur_cls_feats[:, 0], cur_cls_feats[:, 1], c=label2color_100(color_idx), label=cls, edgecolors='none')
    if show_legend:
        ax.legend()   # hide legend
    # ax.grid(True)
    if return_tsne:
        return fig, tsne_results
    else:
        return fig

def tsne_visualize_cmap(features, labels, n_components=2, perplexity=40, n_iter=300, verbose=1, return_tsne=False):
    """
    :param features: input features
    :param labels: feature-wise label
    :param n_components: dimension of embedded space
    :return: TSNE visualization
    """
    fdim = features.shape[-1]
    if fdim > 2:
        tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
        tsne_results = tsne.fit_transform(features)
    else:
        tsne_results = features

    classes = np.unique(labels)

    fig, ax = plt.subplots(figsize=(20, 20))
    # for color_idx, cls in enumerate(classes):
    #     idx = np.argwhere(labels == cls)
    #     idx = np.squeeze(idx)
    #     cur_cls_feats = tsne_results[idx, ...]
    #     cur_cls_feats = np.reshape(cur_cls_feats, [-1, 2])
    #     ax.scatter(cur_cls_feats[:, 0], cur_cls_feats[:, 1], c=color_idx/len(classes), label=cls, edgecolors='none', cmap=plt.get_cmap('winter'))


    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, label=labels, edgecolors='none', cmap=plt.get_cmap('plasma'))
    ax.legend()   # hide legend
    ax.grid(True)
    if return_tsne:
        return fig, tsne_results
    else:
        return fig


def label2color_100(label):
    # num color : 100
    colors = np.array(['tomato', 'dimgray', 'firebrick', 'bisque', 'darkorange',
                       'burlywood', 'forestgreen', 'royalblue', 'slategrey', 'chocolate',
                       'cornflowerblue', 'darkgreen', 'ghostwhite', 'lavender', 'midnightblue',
                       'green', 'grey', 'tab:purple', 'coral', 'mediumseagreen', 'blanchedalmond',
                       'gold', 'blueviolet', 'olive', 'tomato', 'gray', 'lightsteelblue',
                       'aquamarine', 'indianred', 'navajowhite', 'blue', 'darkgoldenrod', 'lawngreen',
                       'lightcoral', 'slateblue', 'brown', 'paleturquoise', 'maroon',
                       'red', 'darkkhaki', 'gold', 'dodgerblue', 'thistle',
                       'indigo', 'tan', 'khaki', 'crimson', 'plum',
                       'violet', 'hotpink', 'seagreen', 'fuchsia', 'orchid', 'deeppink',
                       'purple', 'cadetblue', 'darkviolet', 'pink', 'teal',
                       'cyan', 'palevioletred', 'deepskyblue', 'firebrick', 'palegreen', 'rebeccapurple',
                       'rosybrown', 'yellow', 'antiquewhite', 'steelblue', 'darkseagreen',
                       'peru', 'peachpuff', 'saddlebrown', 'chocolate', 'sienna',
                       'linen', 'tomato', 'salmon', 'yellowgreen', 'lightslategray',
                       'tab:blue', 'tab:orange', 'tab:green', 'azure', 'lime'
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                       'darkgray', 'darkcyan', 'mistyrose', 'orangered', 'navy',
                       'mediumpurple', 'skyblue', 'darksalmon', 'honeydew',
                       'whitesmoke','gainsboro','moccasin','tab:red'

                       ])
    return colors[label]


def label2color_100_old(label):
    # num color : 100
    colors = np.array(['tomato', 'dimgray', 'firebrick', 'bisque', 'darkorange',
                       'burlywood', 'forestgreen', 'darkgreen', 'slategrey', 'lightsteelblue',
                       'cornflowerblue', 'royalblue', 'ghostwhite', 'lavender', 'midnightblue',
                       'green', 'lime', 'seagreen', 'mediumseagreen', 'blanchedalmond',
                       'navajowhite', 'tan', 'antiquewhite', 'gray', 'grey',
                       'aquamarine', 'slateblue', 'blue', 'darkgoldenrod', 'rosybrown',
                       'lightcoral', 'indianred', 'brown', 'firebrick', 'maroon',
                       'red', 'darkkhaki', 'khaki', 'gold', 'paleturquoise',
                       'indigo', 'blueviolet', 'darkviolet', 'thistle', 'plum',
                       'violet', 'purple', 'fuchsia', 'orchid', 'deeppink',
                       'hotpink', 'palevioletred', 'crimson', 'pink', 'teal',
                       'cyan', 'cadetblue', 'deepskyblue', 'steelblue', 'dodgerblue',
                       'lawngreen', 'yellow', 'olive', 'palegreen', 'darkseagreen',
                       'peru', 'peachpuff', 'saddlebrown', 'chocolate', 'sienna',
                       'linen', 'tomato', 'salmon', 'yellowgreen', 'lightslategray',
                       'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                       'darkgray', 'darkcyan', 'mistyrose', 'orangered', 'navy',
                       'mediumpurple', 'skyblue', 'coral', 'darksalmon', 'honeydew',
                       'whitesmoke','gainsboro','moccasin','rebeccapurple', 'azure'

                       ])
    return colors[label]

def label2color_10(label):
    # num color : 10
    colors = np.array(['brown', 'blue', 'green', 'black', 'pink',
                       'grey', 'orange', 'purple', 'yellow', 'red'
                       ])
    return colors[label]