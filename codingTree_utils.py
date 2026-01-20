import heapq
import numba as nb
import numpy as np
import torch
import time
import os

def get_id():
    i = 0
    while True:
        yield i
        i += 1


@nb.jit(nopython=True)
def cut_volume(edge_set, p1, p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            if (p2[j] in edge_set[p1[i]]):
                c12 += edge_set[p1[i]][p2[j]]
    return c12


def merge(new_ID, id1, id2, cut_v, nowhigh, node_dict):
    n1 = node_dict[id1]
    n2 = node_dict[id2]
    new_partition = n1.partition + n2.partition
    v = n1.vol + n2.vol
    g = n1.g + n2.g - 2 * cut_v

    child = set()
    if (n1.children != None):
        child = child.union(n1.children)
    else:
        child.add(n1.ID)
    if (n2.children != None):
        child = child.union(n2.children)
    else:
        child.add(n2.ID)

    new_node = PartitionTreeNode(ID=new_ID, partition=new_partition, high=nowhigh, children=child, g=g, vol=v)

    if (n1.children != None):
        id1_child = n1.children
        for ID in id1_child:
            node_dict[ID].parent = new_ID
        del node_dict[id1]
    else:
        node_dict[id1].parent = new_ID
    if (n2.children != None):
        id2_child = n2.children
        for ID in id2_child:
            node_dict[ID].parent = new_ID
        del node_dict[id2]
    else:
        node_dict[id2].parent = new_ID

    node_dict[new_ID] = new_node


def graph_parse(adj_matrix):
    row = adj_matrix.row
    col = adj_matrix.col
    weight = adj_matrix.data
    g_num_nodes = adj_matrix.shape[0]
    VOL = np.sum(weight)
    node_vol = np.zeros(g_num_nodes)
    Int = nb.types.int32
    Float = nb.types.float64
    ValueDict = nb.types.DictType(Int, Float)
    edge_set = nb.typed.typeddict.Dict.empty(Int, ValueDict)
    adj_table = {}
    edgeNum = 0
    for i in range(g_num_nodes):
        adj = set()
        adj_table[i] = adj
        edge_set[i] = nb.typed.typeddict.Dict.empty(Int, Float)
    for i in range(len(row)):
        if (not col[i] in edge_set[row[i]]):
            edge_set[row[i]][col[i]] = weight[i]
            adj_table[row[i]].add(col[i])
            node_vol[row[i]] += weight[i]
            edgeNum += 1
        if (not row[i] in edge_set[col[i]]):
            edge_set[col[i]][row[i]] = weight[i]
            adj_table[col[i]].add(row[i])
            node_vol[col[i]] += weight[i]
            edgeNum += 1

    print('edge_number: ' + str(edgeNum))
    return g_num_nodes, VOL, node_vol, adj_table, edge_set, edgeNum

def wish_node(edge_set, adj_table, nodes_dict, nodes_ids, g_vol, id):
    node2List = []
    cutvList = []
    v1List = []
    v2List = []
    g1List = []
    g2List = []
    nei = list(adj_table[id])
    n2_id_list = []
    for j in nei:
        n2_id = j
        while nodes_dict[n2_id].parent != None:
            n2_id = nodes_dict[n2_id].parent
        if not n2_id in nodes_ids:
            continue
        n2_id_list.append(n2_id)
        node2List.append(n2_id)
        n1 = nodes_dict[id]
        n2 = nodes_dict[n2_id]
        v1List.append(n1.vol + 1)
        v2List.append(n2.vol + 1)
        g1List.append(n1.g + 1)
        g2List.append(n2.g + 1)
        n1_part = n1.partition
        n2_part = n2.partition
        if len(n1_part) == 1 and len(n2.partition) == 1:
            cut_v = 0
            if (n2_part[0] in edge_set[n1_part[0]]):
                cut_v += edge_set[n1_part[0]][n2_part[0]]
        else:
            cut_v = cut_volume(edge_set, p1=np.array(n1_part), p2=np.array(n2_part))
        cutvList.append(cut_v)
    v1 = np.array(v1List)
    v2 = np.array(v2List)
    g1 = np.array(g1List)
    g2 = np.array(g2List)
    cutvList = np.array(cutvList)
    v12 = v1 + v2
    g12 = g1 + g2 - 2 * cutvList
    diffList = ((v12 - g12) * np.log2(v12) - (v1 - g1) * np.log2(v1) - (v2 - g2) * np.log2(v2) + (
                g12 - g1 - g2) * np.log2(g_vol)) / g_vol
    diffList = list(diffList)
    if len(diffList) == 0:
        return -1,-1,-1
    else:
        index = diffList.index(min(diffList))
        wish_id = n2_id_list[index]
        diff = diffList[index]
        cut_v = cutvList[index]
        return wish_id, diff, cut_v
    
class PartitionTreeNode():
    def __init__(self, ID, partition, vol, g, high=1, children: set = None, parent=None, entropy=0.0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.high = high
        self.entropy = entropy

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

class PartitionTree():
    def __init__(self, adj_matrix):
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table, self.edge_set, self.edgeNum = graph_parse(
            adj_matrix)  
        self.id_g = get_id()
        self.leaves = []
        self.build_leaves()
        self.PartitionTreeEntropy = 0

    def build_leaves(self):
        node_vol = self.node_vol
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            v = node_vol[vertex]
            leaf_node = PartitionTreeNode(ID=ID, partition=[vertex], g=v, vol=v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)

    def __build_k_tree(self, g_vol, nodes_dict: dict, k=2):
        nowhigh = 2
        while (nowhigh <= k):
            if (nowhigh == 2):
                nodes_ids = set(nodes_dict.keys())
                edge_set = self.edge_set
                adj_table = self.adj_table
                for id in nodes_ids:
                    if id in adj_table[id]:
                        adj_table[id].remove(id)
                        del edge_set[id][id]
            else:
                old_new_dict = {}
                new_old_dict = {}
                nodes_ids = []
                nodes_ids_for_return = []
                new_nodes_dict = {}
                Int = nb.types.int32
                Float = nb.types.float64
                ValueDict = nb.types.DictType(Int, Float)
                new_edge_set = nb.typed.typeddict.Dict.empty(Int, ValueDict)
                new_edge_set_for_return = nb.typed.typeddict.Dict.empty(Int, ValueDict)
                adj_table = {}
                for key, value in nodes_dict.items():
                    if (value.parent == None):
                        newID = next(self.id_g)
                        new_leaf_node = PartitionTreeNode(ID=newID, partition=[newID], g=value.g, vol=value.vol,
                                                          high=nowhigh - 1)
                        old_new_dict[key] = newID
                        new_old_dict[newID] = key
                        nodes_ids.append(newID)
                        nodes_ids_for_return.append(key)
                        new_nodes_dict[newID] = new_leaf_node
                nodes_dict.update(new_nodes_dict)
                for i in range(len(nodes_ids)):
                    adj = set()
                    adj_table[nodes_ids[i]] = adj
                    new_edge_set[nodes_ids[i]] = nb.typed.typeddict.Dict.empty(Int, Float)
                    new_edge_set_for_return[nodes_ids_for_return[i]] = nb.typed.typeddict.Dict.empty(Int, Float)
                startIDList = list(edge_set.keys())
                for startID in startIDList:
                    if (nodes_dict[startID].parent != None):
                        startParent_for_return = nodes_dict[startID].parent
                    else:
                        startParent_for_return = startID
                    startParent = old_new_dict[startParent_for_return]
                    weightDict = edge_set[startID]
                    endIDList = list(weightDict.keys())
                    for endID in endIDList:
                        if (nodes_dict[endID].parent != None):
                            endParent_for_return = nodes_dict[endID].parent
                        else:
                            endParent_for_return = endID
                        endParent = old_new_dict[endParent_for_return]
                        weight = weightDict[endID]
                        if (not endParent in adj_table[startParent]):
                            adj_table[startParent].add(endParent)
                        if (endParent in new_edge_set[startParent]):
                            new_edge_set[startParent][endParent] += weight
                            new_edge_set_for_return[startParent_for_return][endParent_for_return] += weight
                        else:
                            new_edge_set[startParent][endParent] = weight
                            new_edge_set_for_return[startParent_for_return][endParent_for_return] = weight
                edge_set = new_edge_set
                nodes_ids = set(nodes_ids)
                for id in nodes_ids:
                    if id in adj_table[id]:
                        adj_table[id].remove(id)
                        del edge_set[id][id]                
            period = 0
            print('start this')
            while period>-1:  
                if period == 1:
                    break
                if len(nodes_ids) == 0:
                    nodes_ids = set()
                    nodes_have_nei = adj_table.keys()
                    for key, value in nodes_dict.items():
                        if value.parent == None and key in nodes_have_nei:
                            nodes_ids.add(key)
                print(len(nodes_ids))
                min_heap = []
                nodes_remove_ids = set()
                for id in nodes_ids:
                    if len(adj_table[id]) == 0:
                        nodes_remove_ids.add(id)
                    else:
                        wish_id, diff, cut_v = wish_node(edge_set, adj_table, nodes_dict, nodes_ids, g_vol, id)
                        if wish_id == -1:
                            nodes_remove_ids.add(id)
                        else:
                            if diff < 0:
                                heapq.heappush(min_heap, (diff, id, wish_id, cut_v))
                            else:
                                nodes_remove_ids.add(id)
                for id in nodes_remove_ids:
                    nodes_ids.remove(id)
                if len(nodes_ids) == 0:
                    period = period + 1
                merged_count = 0
                while merged_count > -1:
                    if len(min_heap) == 0:
                        break
                    diff, id1, id2, cut_v = heapq.heappop(min_heap)
                    if not id1 in nodes_ids or not id2 in nodes_ids:
                        if id1 in nodes_ids and not id2 in nodes_ids:
                            nodes_ids.remove(id1)
                        continue
                    if nodes_dict[id1].merged or nodes_dict[id2].merged:
                        continue
                    nodes_dict[id1].merged = True
                    nodes_dict[id2].merged = True
                    new_id = next(self.id_g)
                    merge(new_id, id1, id2, cut_v, nowhigh, nodes_dict)
                    adj_con = adj_table[id1].union(adj_table[id2])
                    for node in nodes_dict[new_id].partition:
                        if node in adj_con:
                            adj_con.remove(node)                  
                    nodes_ids.remove(id1)
                    nodes_ids.remove(id2)
                    del adj_table[id1]
                    del adj_table[id2]
            if (nowhigh == 2):
                new_nodes_dict = {}
                for key, value in nodes_dict.items():
                    if (value.parent == None and value.high == 1):
                        newID = next(self.id_g)
                        children = set()
                        children.add(key)
                        new_leaf_node = PartitionTreeNode(ID=newID, partition=value.partition, g=value.g, vol=value.vol,
                                                          high=nowhigh, parent=None, children=children)
                        value.parent = newID
                        new_nodes_dict[newID] = new_leaf_node
                nodes_dict.update(new_nodes_dict)               
            else:
                for key, value in nodes_dict.items():
                    if (nodes_dict[key].high == nowhigh):
                        old_child_set = nodes_dict[key].children
                        new_children = set()
                        partition = []
                        for child in old_child_set:
                            new_children.add(new_old_dict[child])
                            partition = partition + nodes_dict[new_old_dict[child]].partition

                        nodes_dict[key].children = new_children
                        nodes_dict[key].partition = partition

                for key, value in old_new_dict.items():
                    if (nodes_dict[value].parent == None):
                        nodes_dict[value].high = nowhigh
                        children = set()
                        children.add(key)
                        nodes_dict[value].children = children
                        nodes_dict[value].partition = nodes_dict[key].partition
                        nodes_dict[key].parent = value

                    else:
                        nodes_dict[key].parent = nodes_dict[value].parent
                        del nodes_dict[value]

                edge_set = new_edge_set_for_return
            nowhigh += 1
        rootID = next(self.id_g)
        rootchild = set()
        for key, value in nodes_dict.items():
            if (value.parent != None):
                continue
            else:
                rootchild.add(key)
                value.parent = rootID
        rootNode = PartitionTreeNode(ID=rootID, partition=[], vol=self.VOL, g=0, children=rootchild, high=nowhigh)
        nodes_dict[rootID] = rootNode

        

        return rootID

    def build_coding_tree(self, k):
        if k == 1:
            print('Error treehigh')
            return
        else:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k)

    def create_node_entropy(self):
        rootID = self.root_id
        IDlist = []
        glist = []
        vlist = []
        v_father_list = []
        VOL = self.VOL
        for k, v in self.tree_node.items():
            if (v.ID == rootID):
                continue
            IDlist.append(v.ID)
            glist.append(v.g)
            vlist.append(v.vol)
            v_father_list.append(self.tree_node[v.parent].vol)
        glist = np.array(glist)
        vlist = np.array(vlist)
        v_father_list = np.array(v_father_list)
        entropyList = -(glist / VOL) * np.log2(vlist / v_father_list)
        for i in range(len(IDlist)):
            if (np.isnan(entropyList[i])):
                self.tree_node[IDlist[i]].entropy = 0.00001
            elif (entropyList[i] == 0.0 or entropyList[i] == -0.0):
                self.tree_node[IDlist[i]].entropy = 0.00001
            else:
                self.tree_node[IDlist[i]].entropy = entropyList[i]

def build_k_coding_tree(dataset, tree_path, matrix, treehigh, K_for_graph_compression=None):
    if K_for_graph_compression == None:
        path = '../data/HG_Data/' + dataset + '/tree/' + tree_path + '/' + str(treehigh) + '.txt'
    else:
        path = '../data/HG_Data/' + dataset + '/tree/' + tree_path + '/H' + str(treehigh) + '_K' + str(K_for_graph_compression) + '.txt'
    if os.path.exists(path):
        f = open(path)
        nodesData = f.readlines()
        tree_node = {}
        for i in range(len(nodesData)):
            if (nodesData[i] == ''):
                break
            nodedata = nodesData[i]
            ID = int(nodedata.split('ID=')[1].split(',partition')[0])
            partitionNodes = nodedata.split('[')[1].split(']')[0]
            if (partitionNodes == ''):
                partition = []
            else:
                partitionNodes = partitionNodes.split(', ')
                partition = [int(j) for j in partitionNodes]
            parent = nodedata.split('parent=')[1].split(',children')[0]
            if (parent == 'None'):
                parent = None
            else:
                parent = int(parent)
            childrenNodes = nodedata.split('children=')[1].split(',vol')[0]
            if (childrenNodes == 'None'):
                children = None
            else:
                childrenNodes = childrenNodes.split('{')[1].split('}')[0].split(', ')
                children = set(int(j) for j in childrenNodes)
            vol = float(nodedata.split('vol=')[1].split(',g')[0])
            g = float(nodedata.split('g=')[1].split(',merged')[0])
            high = int(nodedata.split('high=')[1].split(',entropy')[0])
            entropy = float(nodedata.split('entropy=')[1].split('}')[0])
            node = PartitionTreeNode(ID, partition, vol, g, high, children, parent, entropy)
            tree_node[ID] = node

        partitiontree = PartitionTree(matrix)
        partitiontree.tree_node = tree_node

    else:
        partitiontree = PartitionTree(adj_matrix=matrix)
        partitiontree.build_coding_tree(treehigh)
        partitiontree.create_node_entropy()
        savePartitionTree(partitiontree, path)
    
    return partitiontree

# save node
def savePartitionTree(partitionTree, path):
    with open(path, 'w') as f:
        for k, v in partitionTree.tree_node.items():
            f.write(str(v) + '\n')
    f.close()
