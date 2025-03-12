from copy import deepcopy

import ete3
ete3.__file__
import numpy as np
from matplotlib import cm
from sympy.physics.control.control_plots import matplotlib

from tax_tree_create import create_tax_tree
# from ete3 import NodeStyle, TextFace, add_face_to_node, TreeStyle
import pandas as pd

# matplotlib.use('Qt5Agg')

def creare_tree_view(df_bact_names, family_colors=None):
    """
    Create correlation cladogram, such that tha size of each node is according to the -log(p-value), the color of
    each node represents the sign of the post hoc test, the shape of the node (circle, square,sphere) is based on
    miMic, Utest, or both results accordingly, and if `colorful` is set to True, the background color of the node will be colored based on the family color.
    :param names:  List of sample names (list) :param mean_0: 2D ndarray of the images filled with the post hoc p-values (ndarray).
    :param mean_1:  2D ndarray of the images filled with the post hoc scores (ndarray). :param directory: Folder to
    save the correlation cladogram (str) :param family_colors: Dictionary of family colors (dict) :return: None
    """
    T = ete3.PhyloTree()

    g = create_tax_tree(pd.Series(index=df_bact_names.index))
    epsilon = 1e-1000
    root = list(filter(lambda p: p[1] == 0, g.in_degree))[0][0]
    T.get_tree_root().species = root[0]


    for node in g.nodes:
        for s in g.succ[node]:

            # for u test without mimic results the name is fixed to the correct version of the taxonomy
            # for the mimic results the name is the actual name
            u_test_name = create_list_of_names([(';'.join(s[0]))])[0]

            actual_name = ";".join(s[0])
            if actual_name=='Bacteria;Actinobacteria;Actinobacteria;Bifidobacteriales;Bifidobacteriaceae;Bifidobacterium_0' or actual_name=='Bacteria;Actinobacteria;Actinobacteria;Bifidobacteriales;Bifidobacteriaceae;Bifidobacterium':
                c=0


            if s[0][-1] not in T or not any([anc.species == a for anc, a in
                                             zip(T.search_nodes(name=s[0][-1])[0].get_ancestors()[:-1],
                                                 reversed(s[0]))]):
                t = T
                if len(s[0]) != 1:
                    t = T.search_nodes(full_name=s[0][:-1])[0]

                # nodes in mimic results without u-test

                t = t.add_child(name=s[0][-1])
                t.species = s[0][-1]
                t.add_feature("full_name", s[0])
                name_to_check=create_list_of_names([';'.join(s[0])])[0]
                if name_to_check in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check][0])
                elif name_to_check+';s__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check+';s__'][0])
                elif name_to_check+'_0;s__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check+'_0;s__'][0])
                elif name_to_check + '_0' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check + '_0'][0])
                elif name_to_check+';s__;t__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check+';s__;t__'][0])
                elif name_to_check+'_0;s__;t__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check+'_0;s__;t__'][0])
                elif name_to_check + '_0;t__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check + '_0;t__'][0])
                elif name_to_check + ';t__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check + ';t__'][0])

                else:
                    t.add_feature("max_0_grad",epsilon )

                t.add_feature("shape", "circle")

                if family_colors != None:
                    # setting the family color
                    split_name = actual_name.split(';')
                    if len(split_name) >= 5:
                        family_color = family_colors.get('f__'+actual_name.split(';')[4].split('_')[0], "nocolor")
                    else:
                        family_color = "nocolor"
                    t.add_feature("family_color", family_color)

    T0 = T.copy("deepcopy")
    bound_0 = 0
    for t in T0.get_descendants():
        nstyle = ete3.NodeStyle()
        nstyle["size"] = 30
        nstyle["fgcolor"] = "gray"

        name = ";".join(t.full_name)

        if (t.max_0_grad >bound_0):
            nstyle["fgcolor"] = "blue"
            nstyle["size"] = t.max_0_grad * 100


            nstyle["shape"] = "circle"

            if family_colors != None:
                if t.family_color != "nocolor":
                    # hex_color = rgba_to_hex(t.family_color)
                    nstyle['bgcolor'] = t.family_color

        elif (t.max_0_grad < bound_0):

            nstyle["fgcolor"] = "red"
            nstyle["size"] = t.max_0_grad * 100

            if t.shape == "square":
                nstyle["shape"] = "square"
            if t.shape == "sphere":
                nstyle["shape"] = "sphere"
            if t.shape == "circle":
                nstyle["shape"] = "circle"

            if family_colors != None:
                if t.family_color != "nocolor":
                    # hex_color = rgba_to_hex(t.family_color)
                    nstyle['bgcolor'] = t.family_color

        # if the node is not significant we will still color it by its family color
        if family_colors != None:
            if t.family_color != "nocolor":
                # hex_color = rgba_to_hex(t.family_color)
                nstyle['bgcolor'] = t.family_color

        elif not t.is_root():
            # if the node is not significant, we will detach it
            if not any([anc.max_0_grad > bound_0 for anc in t.get_ancestors()[:-1]]) and not any(
                    [dec.max_0_grad > bound_0 for dec in t.get_descendants()]):
                t.detach()
        t.set_style(nstyle)

    for node in T0.get_descendants():
        if node.is_leaf():
            # checking if the name is ending with _{digit} if so i will remove it
            if node.name[-1].isdigit() and node.name.endswith(f'_{node.name[-1]}'):
                node.name = node.name[:-1]
            name = node.name.replace('_', ' ').capitalize()
            if name == "":
                name = node.get_ancestors()[0].replace("_", " ").capitalize()
            node.name = name

    for node in T0.get_descendants():
        for sis in node.get_sisters():
            siss = []
            if sis.name == node.name:
                node.max_0_grad += sis.max_0_grad
                node.max_1_grad += sis.max_1_grad
                siss.append(sis)
            if len(siss) > 0:
                node.max_0_grad /= (len(sis) + 1)
                node.max_1_grad /= (len(sis) + 1)
                for s in siss:
                    node.remove_sister(s)

    ts = ete3.TreeStyle()
    ts.show_leaf_name = False
    ts.min_leaf_separation = 0.5
    ts.mode = "c"
    ts.root_opening_factor = 0.75
    ts.show_branch_length = False

    D = {1: "(k)", 2: "(p)", 3: "(c)", 4: "(o)", 5: "(f)", 6: "(g)", 7: "(s)",8: "(t)"}

    def my_layout(node):
        """
        Design the cladogram layout.
        :param node: Node ETE object
        :return: None
        """
        #control branch width
        node.img_style["hz_line_width"] = 18
        node.img_style["vt_line_width"] = 18

        if node.is_leaf():
            tax = D[len(node.full_name)]
            if len(node.full_name) == 7:
                name = node.up.name.replace("[", "").replace("]", "") + " " + node.name.lower()
            else:
                name = node.name

            F = ete3.TextFace(f"{name} {tax} ", fsize=100, ftype="Arial")  # {tax}
            ete3.add_face_to_node(F, node, column=0, position="branch-right")

    ts.layout_fn = my_layout
    T0.show(tree_style=(ts))
    # T0.render(f"correlations_tree.svg", tree_style=deepcopy(ts))

def create_list_of_names(list_leaves):
    """
    Fix taxa names for tree plot.
    :param list_leaves: List of leaves names without the initials (list).
    :return: Corrected list taxa names.
    """
    list_lens = [len(i.split(";")) for i in list_leaves]
    otu_train_cols = list()
    for i, j in zip(list_leaves, list_lens):

        if j == 1:
            updated = "k__" + i.split(";")[0]

        elif j == 2:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1]

        elif j == 3:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + \
                      i.split(";")[2]
        elif j == 4:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3]

        elif j == 5:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4]

        elif j == 6:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + \
                      i.split(";")[5]

        elif j == 7:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + i.split(";")[
                          5] + ";" + "s__" + i.split(";")[6]

        elif j == 8:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + i.split(";")[
                          5] + ";" + "s__" + i.split(";")[6] + ";" + "t__" + i.split(";")[7]

        otu_train_cols.append(updated)
    return otu_train_cols


def rgba_to_hex(rgba):
    """
    Convert rgba to hex.
    :param rgba: rgba color (tuple).
    :return: hex color (str).
    """
    return matplotlib.colors.rgb2hex(rgba)


def darken_color(color, factor=0.9):
    return tuple(min(max(comp * factor, 0), 1) for comp in color)

df = pd.read_pickle("/home/finkels9/parkinson/projects_mimic_results_peptibase.pkl")['PRJEB30615']['df_corrs']


# # replace df.index with the match g__ name in the substring
# new_index=[]
# flag=False
# for i in df.index:
#     for count,part in enumerate(i.split(';')):
#         if part[-1].isdigit() and part.endswith(f'_{part[-1]}'):
#             if count==0:
#                 name= part[:-2]+';'+i.split(';')[1]
#                 new_index.append(name)
#                 flag=True
#                 break
#             elif count==1:
#                 name= i.split(';')[0]+';'+part[:-2]
#                 new_index.append(name)
#                 flag=True
#                 break
#
#
#
#     if flag:
#         flag=False
#         continue
#     new_index.append(i)
#
# # df.index=new_index
# df.index = [substring_dict.get('g__'+i.split(';')[0].rsplit('_',4)[2])+';'+i for i in df.index]
#
#
#
cmap_set2 = cm.get_cmap('Set2')
colors_tab10 = [cmap_set2(i) for i in range(cmap_set2.N)]
#
darkened_colors_tab10 = [darken_color(color) for color in colors_tab10]

# darkened_colors_tab10= ['#EADBDD','#FFCAD4','#C9E4DE','#C6DEF1','#DBCF0','#F7F9C4','#E5D0E3','#FFCAE9','#FFF5C1','#FFEDF3']
# extended_colors = darkened_colors_tab10
# # Create a dictionary to store the color for each family
family_colors = {}
#
# # Iterate over unique families and assign colors
names_with_family= [i for i in df.index if len(i.split(';'))>=5]
unique_families = list(set(i.split(';')[4].replace('_0','') for i in names_with_family))
unique_families=set(unique_families)
for i, family in enumerate(unique_families):
    # Use modulo to cycle through the extended color list
    rgba_color = darkened_colors_tab10[i % len(darkened_colors_tab10)]
    # Convert RGBA color to tuple and store it in the dictionary
    family_colors[family] = rgba_color

creare_tree_view(df,family_colors)