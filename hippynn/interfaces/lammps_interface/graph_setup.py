import torch

from ... import IdxType, GraphModule
from ...graphs import get_subgraph, find_relatives, copy_subgraph, find_unique_relative, replace_node
from ...graphs.gops import check_link_consistency
from ...graphs.indextypes import index_type_coercion
from ...graphs.nodes.base import InputNode, AutoNoKw, SingleNode, ExpandParents, MultiNode
from ...graphs.nodes.indexers import PaddingIndexer
from ...graphs.nodes.inputs import SpeciesNode
from ...graphs.nodes.pairs import PairFilter
from ...graphs.nodes.physics import VecMag, GradientNode
from ...graphs.nodes.tags import PairIndexer, Encoder
from ...graphs.nodes.misc import EnsembleTarget

def setup_LAMMPS_graph(energy, extra_properties=None, is_ensemble=False):
    """

    :param energy: energy node for lammp energy_stds interface
    :param is_ensemble: indicates whether ensemble of models is used
    :return: graph for computing from lammps MLIAP unified inputs.
    """
    
    if is_ensemble is True: 
        required_nodes = [energy.mean, energy.std, energy.all] 
    else: 
        required_nodes = [energy]

    if extra_properties is not None:
        model_energy_nodes = extra_properties["model_energy_nodes"]

        required_nodes = required_nodes +model_energy_nodes

    why = "Generating LAMMPS Calculator interface"
    subgraph = get_subgraph(required_nodes)

    search_fn = lambda targ, sg: lambda n: n in sg and isinstance(n, targ)
    pair_indexers = find_relatives(required_nodes, search_fn(PairIndexer, subgraph), why_desc=why)

    new_required, new_subgraph = copy_subgraph(required_nodes, assume_inputed=pair_indexers)
    pair_indexers = find_relatives(new_required, search_fn(PairIndexer, new_subgraph), why_desc=why)

    species = find_unique_relative(new_required, search_fn(SpeciesNode, new_subgraph), why_desc=why)

    encoder = find_unique_relative(species, search_fn(Encoder, new_subgraph), why_desc=why)
    padding_indexer = find_unique_relative(species, search_fn(PaddingIndexer, new_subgraph), why_desc=why)
    inv_real_atoms = padding_indexer.inv_real_atoms

    species_set = torch.as_tensor(encoder.species_set).to(torch.int64)
    min_radius = max(p.dist_hard_max for p in pair_indexers)

    ###############################################################
    # Set up graph to accept external pair indices and shifts

    in_pair_first = InputNode("pair_first")
    in_pair_first._index_state = IdxType.Pair
    in_pair_second = InputNode("pair_second")
    in_pair_second._index_state = IdxType.Pair
    in_pair_coord = InputNode("pair_coord")
    in_pair_coord._index_state = IdxType.Pair
    in_nlocal = InputNode("nlocal")
    in_nlocal._index_state = IdxType.Scalar
    pair_dist = VecMag("pair_dist", in_pair_coord)
    mapped_pair_first = ReIndexAtomNode("pair_first_internal", (in_pair_first, inv_real_atoms))
    mapped_pair_second = ReIndexAtomNode("pair_second_internal", (in_pair_second, inv_real_atoms))

    new_inputs = [species, in_pair_first, in_pair_second, in_pair_coord, in_nlocal]


    # Construct Filters and replace the existing pair indexers with the
    # corresponding new (filtered) node that accepts external pairs of atoms
    for pi in pair_indexers:
        if pi.dist_hard_max == min_radius:
            replace_node(pi.pair_first, mapped_pair_first, disconnect_old=False)
            replace_node(pi.pair_second, mapped_pair_second, disconnect_old=False)
            replace_node(pi.pair_coord, in_pair_coord, disconnect_old=False)
            replace_node(pi.pair_dist, pair_dist, disconnect_old=False)
            pi.disconnect()
        else:
            mapped_node = PairFilter(
                "DistanceFilter-LAMMPS",
                (pair_dist, in_pair_first, in_pair_second, in_pair_coord),
                dist_hard_max=pi.dist_hard_max,
            )
            replace_node(pi.pair_first, mapped_node.pair_first, disconnect_old=False)
            replace_node(pi.pair_second, mapped_node.pair_second, disconnect_old=False)
            replace_node(pi.pair_coord, mapped_node.pair_coord, disconnect_old=False)
            replace_node(pi.pair_dist, mapped_node.pair_dist, disconnect_old=False)
            pi.disconnect()

    energy, energy_stdev,energy_all, *force_model  = new_required
    
    try:
        atom_energies = energy.atom_energies.mean
    except AttributeError:
        atom_energies = energy

    try:
        atom_energies = index_type_coercion(atom_energies, IdxType.Atoms)
    except ValueError:
        raise RuntimeError(
            "Could not build LAMMPS interface. Pass an object with index type IdxType.Atoms or "
            "an object with an `atom_energies` attribute."
        )

    #print("in_nlocal:", in_nlocal.torch_module)
    print("in_pair_first:", in_pair_first)
    print("in_pair_second:", in_pair_second)
    #print("in_pair_first get val:", in_pair_first.get_value())
    print("energy_all:",energy_all)
    if is_ensemble is True:
        local_atom_energy = LocalAtomExtractorNode("local_atom_energy", (atom_energies, in_nlocal))
        local_atom_energy_std = LocalAtomExtractorNode("local_atom_energy_std", (energy_stdev, in_nlocal))
        local_atom_energy_all = LocalAllAtomExtractorNode("local_atom_energy_all", (energy_all, in_nlocal))
    else:    
        local_atom_energy = LocalAtomExtractorNode("local_atom_energy", (atom_energies, in_nlocal))
    grad_rij = GradientNode("grad_rij", (local_atom_energy.total_local_value, in_pair_coord), -1)
   
    print("grad_rij: ", grad_rij)

    # looping over all local sotm energies
    fi_all = []
    grad_rij_all = []
    for i, local_energy in enumerate(force_model): #local_atom_energy_all.total_local_value):
        local_energy = LocalAtomExtractorNode("model_energy", (local_energy, in_nlocal)) 
        grad_r = GradientNode(f"grad_rij_{i}", (local_energy.total_local_value, in_pair_coord), -1)
        grad_rij_all.append(grad_r)

        fi = AtomForceFromPairForceNode(f"atom_force_{i}",parents=(grad_rij_all[i], in_pair_first, in_pair_second, in_nlocal))
        fi_all.append(fi)
    print("grad_rij_all: ", grad_rij_all)
    print("fi_all: ", fi_all)

    atom_force_node_0 = AtomForceFromPairForceNode("atom_force",parents=(grad_rij_all[0], in_pair_first, in_pair_second, in_nlocal))
    print("atom_force_node_0:", atom_force_node_0)

    ensemble_fi = EnsembleTarget("ensemble_fi", fi_all)

    ensemble_fi_all = ensemble_fi.all
    ensemble_fi_std = ensemble_fi.std

    local_atom_force = LocalAtomExtractorNode("local_atom_force", (ensemble_fi_all, in_nlocal))
    local_atom_force_std = LocalAtomExtractorNode("local_atom_force_std", (ensemble_fi_std, in_nlocal))

    print("ensemble_fi:", ensemble_fi)
    print("local_atom_energy.local_atom_values:", local_atom_energy.local_atom_values)
    #fi = atom_force(grad_rij_all, in_pair_coord, in_pair_first, in_pair_second, local_atom_energy.total_local_value )
    implemented_nodes = local_atom_energy.local_atom_values, local_atom_energy.total_local_value, local_atom_energy_std.local_atom_values, grad_rij, local_atom_force.local_atom_values, local_atom_force_std.local_atom_values #ensemble_fi_all, ensemble_fi_std #ensemble_fi#grad_rij #grad_rij_all

    check_link_consistency((*new_inputs, *implemented_nodes))
    mod = GraphModule(new_inputs, implemented_nodes)
    mod.eval()

    return min_radius / 2, species_set, mod

'''class AtomicForceNode():
    def forward(self, data, fij):
        for ii in range(data.npair:
            ii3 = ii*3 
            i = data.pair_i[ii]
            j = data.jatoms[ii]

            f[i][0] +=fij[ii3]
            f[i][1] +=fij[ii3+1]
            f[i][2] +=fij[ii3+2]

            
            f[j][0] -=fij[ii3]
            f[j][1] -=fij[ii3+1]
            f[j][2] -=fij[ii3+2]

    return f'''

#lass AtomicForce():
def atom_force(fij, in_pair_coord, in_pair_first, in_pair_second,local_atoms ):
        
    f = torch.zeros((256, 3))
    for i in range(len(fij)):
        a1 = in_pair_first.get_value(i)
        print("a1:", a1)
        a2 = in_pair_seond.get_value(i)
        print("a2:", a2)

        f[a1][0] +=fij[ii3]
        f[a1][1] +=fij[ii3+1]
        f[a1][2] +=fij[ii3+2]

            
        f[a2][0] -=fij[ii3]
        f[a2][1] -=fij[ii3+1]
        f[a2][2] -=fij[ii3+2]
    return f

class AtomForceFromPairForce(torch.nn.Module):
    def forward(self, f_ij, in_pair_first, in_pair_second, in_nlocal):
        #in_nlocal_cpu = in_nlocal.detach().cpu().item()
        #print("in_nlocal:", in_nlocal)
        #val = in_nlocal.item()
        #print(val, type(val))
        f_i = torch.zeros((256, 3), device=f_ij.device, dtype=f_ij.dtype)
        #print("in_pair_first", in_pair_first) 
        f_i.index_add(0, in_pair_first, f_ij)
        f_i.index_add(0, in_pair_second, -f_ij)

        return f_i

class AtomForceFromPairForceNode(AutoNoKw, SingleNode): # ExpandParents, MultiNode):
    _input_names = "f_ij", "in_pair_first", "in_pair_second", "in_nlocal"
    _output_names = "f_i"
    _main_output = "f_i"
    _output_index_states = (IdxType.Atoms,)
    _auto_module_class = AtomForceFromPairForce

    #_parent_expander.assertlen(2)
    #_parent_expander.get_main_outputs()
    #_parent_expander.require_idx_states(IdxType.Atoms)
    def __init__(self, name, parents, module="auto", **kwargs):
        super().__init__(name, parents, module=module, **kwargs)
        self._index_state = IdxType.Atoms


class ReIndexAtomMod(torch.nn.Module):
    def forward(self, raw_atom_index_array, inverse_real_atoms):
        return inverse_real_atoms[raw_atom_index_array]


class ReIndexAtomNode(AutoNoKw, SingleNode):
    _input_names = "raw_atom_index_array", "inverse_real_atoms"
    _main_output = "total_local_energy"
    _auto_module_class = ReIndexAtomMod

    def __init__(self, name, parents, module="auto", **kwargs):
        self._index_state = parents[0]._index_state
        super().__init__(name, parents, module=module, **kwargs)


class LocalAllAtomExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, all_atom_values, nlocal):
        local_atom_values = all_atom_values[:, :nlocal]  # [n_models, nlocal]
        total_local_value = local_atom_values.sum(dim=1)
        return local_atom_values, total_local_value

class LocalAllAtomExtractorNode(AutoNoKw, ExpandParents, MultiNode):
    _input_names = "all_atom_values", "nlocal"  
    _output_names = "local_atom_values", "total_local_value" 
    _main_output = "total_local_value"
    _output_index_states = None, IdxType.Scalar 
    _auto_module_class = LocalAllAtomExtractor

    _parent_expander.assertlen(2)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, IdxType.Scalar )

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)

    #def __iter__(self):
        #for attr in dir(self):
        #    if not attr.startswith("__"):
        #        yield attr

class LocalAtomExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, all_atom_values, nlocal):
        #if torch.is_tensor(nlocal):
        #    nlocal = nlocal.item()
        local_atom_values = all_atom_values[:nlocal]
        total_local_value = torch.sum(local_atom_values)
        return local_atom_values, total_local_value


class LocalAtomExtractorNode(AutoNoKw, ExpandParents, MultiNode):
    _input_names = "all_atom_values", "nlocal"  
    _output_names = "local_atom_values", "total_local_value" 
    _main_output = "total_local_value"
    _output_index_states = None, IdxType.Scalar 
    _auto_module_class = LocalAtomExtractor

    _parent_expander.assertlen(2)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, IdxType.Scalar )

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)
