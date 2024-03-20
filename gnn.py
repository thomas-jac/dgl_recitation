import torch.nn as nn
import torch
import dgl
import dgl.function as dglfn

class MoleculeRegressor(nn.Module):
    def __init__(self, n_node_feats: int, n_edge_feats: int, hidden_node_feats: int = 64, n_convs: int = 3):
        super().__init__()
        
        self.input_node_feats = n_node_feats
        self.input_edge_feats = n_edge_feats
        self.hidden_node_feats = hidden_node_feats
        self.n_convs = n_convs


        self.conv_layers = []
        for i in range(n_convs):

            input_edge_feats = n_edge_feats
            output_node_feats = hidden_node_feats

            if i == 0:
                input_node_feats = n_node_feats
            else:
                input_node_feats = hidden_node_feats

            self.conv_layers.append(MoleculeConvolution(input_node_feats, input_edge_feats, output_node_feats))
        
        self.conv_layers = nn.ModuleList(self.conv_layers)


        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_node_feats, hidden_node_feats//2),
            nn.SiLU(),
            nn.Linear(hidden_node_feats//2, 1)
        )

    def forward(self, g):
        
        with g.local_scope():
            node_feats = g.ndata['attr']
            for conv_layer in self.conv_layers:
                node_feats = conv_layer(g)
                g.ndata['attr'] = node_feats
            
            # pool all node features
            graph_feats = dgl.readout_nodes(g, 'attr', op='sum')
            graph_feats = graph_feats / 10

            predictions = self.prediction_head(graph_feats)
        
        return predictions



class MoleculeConvolution(nn.Module):

    def __init__(self, input_node_feats: int, input_edge_feats: int, output_node_feats: int = 64,):

        super().__init__()

        self.input_node_feats = input_node_feats
        self.input_edge_feats = input_edge_feats
        self.output_node_feats = output_node_feats

        self.message_fn = nn.Sequential(
            nn.Linear(input_node_feats*2 + input_edge_feats + 1, output_node_feats),
            nn.SiLU()
        )
        
        
        self.update_fn = nn.Sequential(
            nn.Linear(output_node_feats, output_node_feats),
            nn.SiLU(),
            nn.LayerNorm(output_node_feats)
        )

    def forward(self, g: dgl.DGLGraph):

        with g.local_scope():

            # compute messages along every edge and aggregate them
            g.update_all(message_func=self.message, reduce_func=dgl.function.sum('m', 'm_agg'))

            # update node states
            input_node_features = g.ndata['attr']
            updated_node_feats = self.update_fn(g.ndata['m_agg']/10)

        return updated_node_feats

    def message(self, edges):


        message_inputs = []

        # add source and destination node features to message inputs
        message_inputs.append(edges.src['attr'])
        message_inputs.append(edges.dst['attr'])

        # compute distance along every edge, add to message inputs
        x_diff = edges.src['pos'] - edges.dst['pos']
        dij = _norm_no_nan(x_diff, keepdims=True)
        message_inputs.append(dij)

        # add edge type to message inputs
        message_inputs.append(edges.data['edge_attr'])

        # concatenate message inputs
        message_inputs = torch.cat(message_inputs, dim=-1)

        # compute messages
        m = self.message_fn(message_inputs)

        # return messages
        return {'m': m}




def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out
