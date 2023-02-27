import os
import numpy as np
import parmed
from pyrosetta import Pose
from math import pi as _pi
import utils.protein_distance as pro_dist
import math
import time

class Charmm(object):
    def __init__(self, parameters_file_path, topology_file_path, save_path):
        self.params_dict = self.read_charmm_parameters_file(parameters_file_path)
        self.top_dict = self.read_charmm_topology_file(topology_file_path)

        # if isinstance(pdb, Pose):
        #     self.pdb = parmed.load_rosetta(pdb)
        # else:
        #     self.pdb = parmed.load_file(pdb)

        '''
        file_atoms = parmed.load_file(save_path).coordinates
        file_atoms = np.around(file_atoms, decimals=3).tolist()
        pose_atoms = parmed.load_rosetta(pose).coordinates
        pose_atoms = np.around(pose_atoms, decimals=3).tolist()

        self.convert_index = []
        for file_atoms_index, x in enumerate(file_atoms):
            if x in pose_atoms:
                self.convert_index.append(pose_atoms.index(x))
        '''
        self.pdb = parmed.load_file(save_path)

        self.res_name_list, self.res_index_list = self.get_residues_name_list(self.pdb)
        self.analysis_pdb_struct()


    def analysis_pdb_struct(self):
        self.bond_mat, charmm_name_atom_struct = self.get_bond_elem(self.pdb, self.top_dict, self.params_dict)
        atom_index_dict = {}
        for x in self.bond_mat:
            atom1, atom2 = list(x[0])
            if atom1 not in atom_index_dict:
                atom_index_dict[atom1] = [atom2]
            else:
                atom_index_dict[atom1].append(atom2)
            if atom2 not in atom_index_dict:
                atom_index_dict[atom2] = [atom1]
            else:
                atom_index_dict[atom2].append(atom1)
        self.angle_mat, self.angle_ub_mat = self.get_angle_elem(atom_index_dict, charmm_name_atom_struct, self.res_name_list,
                                                                self.top_dict, self.params_dict)
        self.dihedral_mat, self.improper_mat = self.get_dihedral_elem(atom_index_dict, self.res_index_list,
                                                                      charmm_name_atom_struct, self.res_name_list,
                                                                      self.top_dict, self.params_dict)

    def calculate_unit(self, calc_pdb):
        #atoms_coords = parmed.load_rosetta(calc_pdb).coordinates[self.convert_index]
        atoms_coords = parmed.load_file(calc_pdb).coordinates
        DEG_TO_RAD = np.pi / 180.0

        # bond unit
        energy_bond = 0
        for x in self.bond_mat:
            atoms = list(x[0])
            atom1_index, atom2_index = atoms
            atoms1_coord = atoms_coords[atom1_index]
            atoms2_coord = atoms_coords[atom2_index]
            d = np.sqrt(pro_dist.distance2(atoms1_coord, atoms2_coord))
            #atom1, atom2 = atoms_list[atom1_index], atoms_list[atom2_index]
            #bond = parmed.Bond(atom1, atom2)
            bond_type = x[1]
            #d = bond.measure()
            dx = d - bond_type[1]
            energy_item = bond_type[0] * dx * dx
            energy_bond = energy_bond + energy_item

        # angle unit
        energy_angle = 0
        for x in self.angle_mat:
            atoms = list(x[0])
            atom1_index, atom2_index, atom3_index = atoms
            atoms1_coord = atoms_coords[atom1_index]
            atoms2_coord = atoms_coords[atom2_index]
            atoms3_coord = atoms_coords[atom3_index]

            angle_type = x[1]
            #angle = parmed.Angle(atom1, atom2, atom3)
            #theta = angle.measure()
            theta = pro_dist.angle(atoms1_coord, atoms2_coord, atoms3_coord)
            d_theta = (theta - angle_type[1]) * DEG_TO_RAD
            energy_item = d_theta * d_theta * angle_type[0]
            energy_angle = energy_angle + energy_item

        # ub unit
        energy_ub = 0
        for x in self.angle_ub_mat:
            atoms = list(x[0])
            atom1_index, _, atom3_index = atoms
            atoms1_coord = atoms_coords[atom1_index]
            atoms3_coord = atoms_coords[atom3_index]
            ub_type = x[1]

            s = np.sqrt(pro_dist.distance2(atoms1_coord, atoms3_coord))

            #atom1, atom3 = atoms_list[atom1_index], atoms_list[atom3_index]

            #bond = parmed.Bond(atom1, atom3)

            #s = bond.measure()
            ds = (s - ub_type[1])
            energy_item = ds * ds * ub_type[0]
            energy_ub = energy_ub + energy_item

        # dihedral unit
        energy_dihedral = 0
        for x in self.dihedral_mat:
            atoms = list(x[0])
            atom1_index, atom2_index, atom3_index, atom4_index = atoms
            atoms1_coord = atoms_coords[atom1_index]
            atoms2_coord = atoms_coords[atom2_index]
            atoms3_coord = atoms_coords[atom3_index]
            atoms4_coord = atoms_coords[atom4_index]
            chi = pro_dist.dihedral(atoms1_coord, atoms2_coord, atoms3_coord, atoms4_coord) * DEG_TO_RAD

            #atom1, atom2, atom3, atom4 = atoms_list[atom1_index], atoms_list[atom2_index], atoms_list[atom3_index], \
            #                             atoms_list[atom4_index]
            #dihedral = parmed.Dihedral(atom1, atom2, atom3, atom4)
            #chi = dihedral.measure() * DEG_TO_RAD
            dihedral_type = x[1]
            if len(dihedral_type) == 1:
                energy_item = dihedral_type[0][0] * (
                            1 + np.cos(dihedral_type[0][1] * chi - dihedral_type[0][2] * DEG_TO_RAD))
            else:
                energy_item = 0
                for y in dihedral_type:
                    energy_item += y[0] * (1 + np.cos(y[1] * chi - y[2] * DEG_TO_RAD))
            energy_dihedral = energy_item + energy_dihedral

        # improper unit
        energy_improper = 0
        for x in self.improper_mat:
            atoms = list(x[0])
            atom1_index, atom2_index, atom3_index, atom4_index = atoms
            #atom1, atom2, atom3, atom4 = atoms_list[atom1_index], atoms_list[atom2_index], atoms_list[atom3_index], \
            #                             atoms_list[atom4_index]
            atoms1_coord = atoms_coords[atom1_index]
            atoms2_coord = atoms_coords[atom2_index]
            atoms3_coord = atoms_coords[atom3_index]
            atoms4_coord = atoms_coords[atom4_index]
            phi = pro_dist.dihedral(atoms1_coord, atoms2_coord, atoms3_coord, atoms4_coord)
            improper_type = x[1]

            #improper = parmed.Improper(atom1, atom2, atom3, atom4)
            #phi = improper.measure()
            dphi = (improper_type[1] - phi) * DEG_TO_RAD

            energy_item = improper_type[0] * dphi * dphi

            energy_improper = energy_item + energy_improper

        return energy_bond + energy_angle + energy_ub + energy_dihedral + energy_improper


    def get_charmm_name(self, atom_index, top_dict, charmm_name_atom_struct, res_name_list):
        atom_name_key = '{}_{}'.format(res_name_list[atom_index], charmm_name_atom_struct[atom_index])
        if atom_name_key in top_dict['atom']:
            atom_name = top_dict['atom'][atom_name_key]
        else:
            atom_name = charmm_name_atom_struct[atom_index]
        return atom_name


    def modify_atom_name(self, atom_name, res_name, first_flag):
        if res_name == 'ILE' and 'HD1' in atom_name:
            atom_name = atom_name.replace('HD1', 'HD')
        if res_name == 'ILE' and 'CD1' in atom_name:
            atom_name = atom_name.replace('CD1', 'CD')
        if res_name == 'SER' and 'HG' in atom_name:
            atom_name = atom_name.replace('HG', 'HG1')
        if atom_name == 'OXT':
            atom_name = 'OC'

        if atom_name[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            atom_name = atom_name[1:] + atom_name[0]
        else:
            atom_name = atom_name

        if first_flag:
            if atom_name == 'N':
                atom_name = 'NH3'
            if atom_name == 'H1':
                atom_name = 'HC'
            if atom_name == 'H2':
                atom_name = 'HC'
            if atom_name == 'H3':
                atom_name = 'HC'
        if atom_name == 'H':
            atom_name = 'HN'
        return atom_name

    def read_charmm_topology_file(self, file_path):
        res_name = ''
        charmm_bond = {}
        charmm_atom_dict = {}
        charmm_improper = {}

        for x in open(file_path, 'r'):
            if x == '\n':
                res_name = ''
            if '!' in x:
                line = x[:x.index('!')]
            else:
                line = x.replace('\n', '')
            if ('RESI' in x or 'PRES' in x) and res_name == '':
                res_name = line.split()[1]
                if res_name == 'HSP':
                    res_name = 'HIS'
                charmm_bond[res_name] = []
                charmm_improper[res_name] = []
            if res_name != '':

                if 'BOND' in line or 'DOUBLE' in line:
                    atom_pair = line.split()[1:]

                    for atom_index in range(int(len(atom_pair) / 2)):
                        charmm_bond[res_name].append({atom_pair[atom_index * 2], atom_pair[atom_index * 2 + 1]})
                if 'ATOM' in line:
                    atom_data = line.split()
                    pdb_atom_name = atom_data[1]
                    charmm_atom_type = atom_data[2]
                    dict_key = '{}_{}'.format(res_name, pdb_atom_name)
                    charmm_atom_dict[dict_key] = charmm_atom_type
                if 'IMPR' in line:
                    atom_pair = line.split()[1:]
                    for atom_index in range(int(len(atom_pair) / 4)):
                        charmm_improper[res_name].append([atom_pair[atom_index * 4], atom_pair[atom_index * 4 + 1],
                                                          atom_pair[atom_index * 4 + 2], atom_pair[atom_index * 4 + 3]])
        charmm_top = {
            'bond': charmm_bond,
            'atom': charmm_atom_dict,
            'impr': charmm_improper
        }

        return charmm_top

    def read_charmm_parameters_file(self, file_path):
        bond_block = False
        angle_block = False
        dihedrals_block = False
        improper_block = False
        cmap_block = False
        nonbonded_block = False

        LJ_params = {}
        LJ_params_14 = {}
        camp_params = {}
        improper_params = {}
        dihedral_params = {}
        angle_params = {}
        angle_UB_params = {}
        bond_params = {}

        for x in open(file_path, 'r'):

            if '!' in x:
                line = x[:x.index('!')]
            else:
                line = x.replace('\n', '')
            if x[0] == '*':
                continue
            if line == '':
                continue

            if 'BONDS' in line:
                bond_block = True
                continue
            if 'ANGLES' in line:
                angle_block = True
                continue
            if 'DIHEDRALS' in line:
                dihedrals_block = True
                continue
            if 'IMPROPER' in line:
                improper_block = True
                continue
            if 'CMAP' in line:
                cmap_block = True
                continue
            if 'NONBONDED' in line:
                nonbonded_block = True
                continue
            if 'NBFIX' in line:
                break

            if nonbonded_block:
                # V(Lennard - Jones) = Eps, i, j[(Rmin, i, j / ri, j) ** 12 - 2(Rmin, i, j / ri, j) ** 6]
                nonbonded_data = line.split()
                if nonbonded_data == []:
                    continue

                if len(nonbonded_data) == 4:
                    nonbonded_key = nonbonded_data[0]
                    ignored, epsilon, rmin = nonbonded_data[1:]
                    LJ_params[nonbonded_key] = np.array([ignored, epsilon, rmin], dtype=float)
                if len(nonbonded_data) == 7:
                    nonbonded_key = nonbonded_data[0]
                    ignored, epsilon, rmin, ignored_14, epsilon_14, rmin_14 = nonbonded_data[1:]
                    LJ_params_14[nonbonded_key] = np.array([ignored, epsilon, rmin, ignored_14, epsilon_14, rmin_14],
                                                           dtype=float)
            elif cmap_block:
                continue
            elif improper_block:
                # V(improper) = kpsi(psi - psi0)**2
                improper_data = line.split()
                if improper_data == []:
                    continue

                improper_key = '{} {} {} {}'.format(*improper_data[:4])
                kpsi, psi0 = improper_data[4], improper_data[6]
                improper_params[improper_key] = np.array([kpsi, psi0], dtype=float)
            elif dihedrals_block:
                # V(dihedral) = kchi(1 + cos(n(chi) - delta))
                dihedral_data = line.split()
                if dihedral_data == []:
                    continue

                dihedral_key = '{} {} {} {}'.format(*dihedral_data[:4])
                kchi, n, delta = dihedral_data[4:]
                if dihedral_key not in dihedral_params:
                    dihedral_params[dihedral_key] = []
                dihedral_params[dihedral_key].append(np.array([kchi, n, delta], dtype=float).tolist())
            elif angle_block:
                # V(angle) = Ktheta(Theta - Theta0) ** 2
                # V(Urey_Bradley) = Kub(S - S0) ** 2
                angle_data = line.split()
                if angle_data == []:
                    continue

                angle_key = '{} {} {}'.format(*angle_data[:3])
                ktheta, theta0 = angle_data[3: 5]
                angle_params[angle_key] = np.array([ktheta, theta0], dtype=float)
                if len(angle_data) == 7:
                    kub, s0 = angle_data[5:]
                    angle_UB_params[angle_key] = np.array([kub, s0], dtype=float)
            elif bond_block:
                # V(bond) = Kb(b - b0)**2
                bond_data = line.split()
                if bond_data == []:
                    continue

                bond_key = '{} {}'.format(*bond_data[:2])
                kb, b0 = bond_data[2:]
                bond_params[bond_key] = np.array([kb, b0], dtype=float)

        charmm_params = {
            'BOND': bond_params,
            'ANGLE': angle_params,
            'UB': angle_UB_params,
            'DIHEDRAL': dihedral_params,
            'IMPROPER': improper_params,
            'CMAP': camp_params,
            'LJ': LJ_params,
            'LJ_14': LJ_params_14
        }

        return charmm_params

    def get_residues_name_list(self, pdb):
        name_list = []
        index_list = []
        for index, res in enumerate(pdb.residues):
            name_list = name_list + [res.name] * len(res.atoms)
            index_list = index_list + [index] * len(res.atoms)
        return name_list, index_list

    def get_bond_elem(self, pdb, top_dict, param_dict):
        res_num = len(pdb.residues)
        first_res_flag = True
        last_res_flag = False
        count = 0

        charmm_name_atom_struct = []
        res_struct = []

        for res_index, res in enumerate(pdb.residues):
            if res_index == res_num - 1:
                last_res_flag = True

            res_atom_list = []
            res_modify_index = {}
            for atom_index, atom in enumerate(res):
                atom_name = self.modify_atom_name(atom.name, res.name, first_res_flag)
                if last_res_flag:
                    if atom_index == 2:
                        atom_name = 'CC'
                    if atom_index == 3:
                        atom_name = 'OC'

                key = '{}_{}'.format(res.name, atom_name)
                res_atom_list.append(atom_name)
                if key in top_dict['atom']:
                    atom.name = top_dict['atom'][key]
                else:
                    #print(res.name, atom_name)
                    atom.name = atom_name
            first_res_flag = False

            charmm_name_atom_struct = charmm_name_atom_struct + res_atom_list
            res_struct.append([res.name, len(res_atom_list)])

        # bond_mat = np.zeros((len(charmm_name_atom_struct), len(charmm_name_atom_struct)), dtype=bool)
        bond_mat = []

        atom_count = 0
        first_res_flag = True
        last_res_flag = False
        for res_index, res in enumerate(res_struct):
            res_name = res[0]
            res_bond_top = top_dict['bond'][res_name].copy()
            res_atom_list = charmm_name_atom_struct[atom_count: res[1] + atom_count]
            if res_index == res_num - 1:
                last_res_flag = True
                res_bond_top.remove({'C', '+N'})
                res_bond_top.append({'OC', 'CC'})
                res_bond_top.append({'CA', 'CC'})
            if first_res_flag:
                res_bond_top.append({'NH3', 'HC'})
            first_res_flag = False

            for bond in res_bond_top:
                atom1, atom2 = list(bond)
                if atom1 == '+N':
                    atom1_index = [atom_count + res[1]]
                    atom2_index = [i + atom_count for i, x in enumerate(res_atom_list) if x == atom2]
                    atom1 = 'N'
                    # print('atom1', atom1, atom2, atom_list[atom1_index[0]-1], atom_list[atom2_index[0]])
                    atom1_charmm_name_key = '{}_{}'.format(res_struct[res_index + 1][0], atom1)
                    atom2_charmm_name_key = '{}_{}'.format(res_name, atom2)

                elif atom2 == '+N':
                    atom2_index = [atom_count + res[1]]
                    atom1_index = [i + atom_count for i, x in enumerate(res_atom_list) if x == atom1]
                    atom2 = 'N'
                    # print('atom2', atom1, atom2, atom_list[atom1_index[0]], atom_list[atom2_index[0]-1])
                    atom1_charmm_name_key = '{}_{}'.format(res_name, atom1)
                    atom2_charmm_name_key = '{}_{}'.format(res_struct[res_index + 1][0], atom2)

                else:
                    atom1_index = [i + atom_count for i, x in enumerate(res_atom_list) if x == atom1]
                    atom2_index = [i + atom_count for i, x in enumerate(res_atom_list) if x == atom2]
                    atom1_charmm_name_key = '{}_{}'.format(res_name, atom1)
                    atom2_charmm_name_key = '{}_{}'.format(res_name, atom2)

                atom1_charmm_name = top_dict['atom'][atom1_charmm_name_key] if atom1_charmm_name_key in top_dict[
                    'atom'] else atom1
                atom2_charmm_name = top_dict['atom'][atom2_charmm_name_key] if atom2_charmm_name_key in top_dict[
                    'atom'] else atom2
                bond_name = '{} {}'.format(atom1_charmm_name, atom2_charmm_name)
                if bond_name in param_dict['BOND']:
                    bond_type = param_dict['BOND'][bond_name]
                else:
                    bond_name = '{} {}'.format(atom2_charmm_name, atom1_charmm_name)
                    if bond_name in param_dict['BOND']:
                        bond_type = param_dict['BOND'][bond_name]
                    else:
                        continue
                for atom1_seq_index in atom1_index:
                    for atom2_seq_index in atom2_index:
                        bond_mat.append([{atom1_seq_index, atom2_seq_index}, bond_type])
            atom_count = atom_count + res[1]

        first_bond_type = param_dict['BOND']['NH3 CT1']
        bond_mat.append(([{0, 1}, first_bond_type]))
        return bond_mat, charmm_name_atom_struct

    def get_angle_elem(self, atom_index_dict, charmm_name_atom_struct, res_name_list, top_dict, params_dict):
        angle_mat = []
        angle_ub_mat = []
        for atom_mid in atom_index_dict:
            atom_mid_charmm_name_key = '{}_{}'.format(res_name_list[atom_mid], charmm_name_atom_struct[atom_mid])
            if atom_mid_charmm_name_key in top_dict['atom']:
                atom_mid_charmm_name = top_dict['atom'][atom_mid_charmm_name_key]
            else:
                atom_mid_charmm_name = charmm_name_atom_struct[atom_mid]

            links_atoms = atom_index_dict[atom_mid]
            for atom_1 in links_atoms:
                atom_1_charmm_name_key = '{}_{}'.format(res_name_list[atom_1], charmm_name_atom_struct[atom_1])
                if atom_1_charmm_name_key in top_dict['atom']:
                    atom_1_charmm_name = top_dict['atom'][atom_1_charmm_name_key]
                else:
                    atom_1_charmm_name = charmm_name_atom_struct[atom_1]

                for atom_2 in links_atoms:
                    if atom_1 == atom_2:
                        continue
                    atom_2_charmm_name_key = '{}_{}'.format(res_name_list[atom_2], charmm_name_atom_struct[atom_2])
                    if atom_2_charmm_name_key in top_dict['atom']:
                        atom_2_charmm_name = top_dict['atom'][atom_2_charmm_name_key]
                    else:
                        atom_2_charmm_name = charmm_name_atom_struct[atom_2]
                    angle_key = '{} {} {}'.format(atom_1_charmm_name, atom_mid_charmm_name, atom_2_charmm_name)
                    if angle_key in params_dict['ANGLE']:
                        angle_type = params_dict['ANGLE'][angle_key]
                        if [[atom_2, atom_mid, atom_1], angle_type] not in angle_mat:
                            angle_mat.append([[atom_1, atom_mid, atom_2], angle_type])
                    if angle_key in params_dict['UB']:
                        angle_ub_type = params_dict['UB'][angle_key]
                        if [[atom_2, atom_mid, atom_1], angle_ub_type] not in angle_ub_mat:
                            angle_ub_mat.append([[atom_1, atom_mid, atom_2], angle_ub_type])

        angle_mat = self.get_unique_list(angle_mat)
        angle_ub_mat = self.get_unique_list(angle_ub_mat)

        return angle_mat, angle_ub_mat

    def get_dihedral_elem(self, atom_index_dict, res_index_list, charmm_name_atom_struct, res_name_list, top_dict,
                          params_dict):
        dihedral_mat = []
        dihedral_pos_mat = []
        improper_mat = []
        improper_mat_X = []
        for atom1_index in atom_index_dict:
            atom1_name = self.get_charmm_name(atom1_index, top_dict, charmm_name_atom_struct, res_name_list)
            atom1_link = atom_index_dict[atom1_index]
            for atom2_index in atom1_link:
                if atom1_index == atom2_index:
                    continue
                atom2_name = self.get_charmm_name(atom2_index, top_dict, charmm_name_atom_struct, res_name_list)
                atom2_link = atom_index_dict[atom2_index]
                for atom3_index in atom2_link:
                    if atom3_index in [atom1_index, atom2_index]:
                        continue
                    atom3_name = self.get_charmm_name(atom3_index, top_dict, charmm_name_atom_struct, res_name_list)
                    atom3_link = atom_index_dict[atom3_index]
                    for atom4_index in atom3_link:
                        if atom4_index in [atom1_index, atom2_index, atom3_index]:
                            continue
                        atom4_name = self.get_charmm_name(atom4_index, top_dict, charmm_name_atom_struct, res_name_list)
                        dihedral_key = '{} {} {} {}'.format(atom1_name, atom2_name, atom3_name, atom4_name)
                        if dihedral_key in params_dict['DIHEDRAL']:
                            dihedral_type = params_dict['DIHEDRAL'][dihedral_key]
                            if [atom4_index, atom3_index, atom2_index, atom1_index] not in dihedral_pos_mat:
                                dihedral_mat.append(
                                    [[atom1_index, atom2_index, atom3_index, atom4_index], dihedral_type])
                                dihedral_pos_mat.append([atom1_index, atom2_index, atom3_index, atom4_index])

        for atom1_index in atom_index_dict:
            atom1_name = self.get_charmm_name(atom1_index, top_dict, charmm_name_atom_struct, res_name_list)
            atom1_link = atom_index_dict[atom1_index]
            for atom2_index in atom1_link:
                if atom1_index == atom2_index:
                    continue
                atom2_name = self.get_charmm_name(atom2_index, top_dict, charmm_name_atom_struct, res_name_list)
                atom2_link = atom_index_dict[atom2_index]
                for atom3_index in atom2_link:
                    if atom3_index in [atom1_index, atom2_index]:
                        continue
                    atom3_name = self.get_charmm_name(atom3_index, top_dict, charmm_name_atom_struct, res_name_list)
                    atom3_link = atom_index_dict[atom3_index]
                    for atom4_index in atom3_link:
                        if atom4_index in [atom1_index, atom2_index, atom3_index]:
                            continue
                        atom4_name = self.get_charmm_name(atom4_index, top_dict, charmm_name_atom_struct, res_name_list)
                        dihedral_key_X_mid = 'X {} {} X'.format(atom2_name, atom3_name)
                        dihedral_key_X_side = '{} X X {}'.format(atom1_name, atom4_name)
                        if dihedral_key_X_mid in params_dict['DIHEDRAL']:
                            dihedral_type = params_dict['DIHEDRAL'][dihedral_key_X_mid]
                            if [atom1_index, atom2_index, atom3_index, atom4_index] not in dihedral_pos_mat and [
                                atom4_index, atom3_index, atom2_index, atom1_index] not in dihedral_pos_mat:
                                dihedral_mat.append(
                                    [[atom1_index, atom2_index, atom3_index, atom4_index], dihedral_type])
                                dihedral_pos_mat.append([atom1_index, atom2_index, atom3_index, atom4_index])
                                # print(res_name_list[atom1_index], dihedral_key_X_mid)
                        elif dihedral_key_X_side in params_dict['DIHEDRAL']:
                            dihedral_type = params_dict['DIHEDRAL'][dihedral_key_X_side]
                            if [atom1_index, atom2_index, atom3_index, atom4_index] not in dihedral_pos_mat and [
                                atom4_index, atom3_index, atom2_index, atom1_index] not in dihedral_pos_mat:
                                dihedral_mat.append(
                                    [[atom1_index, atom2_index, atom3_index, atom4_index], dihedral_type])
                                dihedral_pos_mat.append([atom1_index, atom2_index, atom3_index, atom4_index])

        improper_pos_noX_mat = {}
        improper_pos_X_mat = {}  # key is 'first_atom_index last_atom_index', value is {atom_mid_1, atom_mid_2}
        for atom3_index in atom_index_dict:
            # atom3 is first atom
            atom3_name = self.get_charmm_name(atom3_index, top_dict, charmm_name_atom_struct, res_name_list)
            atom3_link = atom_index_dict[atom3_index]

            for atom1_index in atom3_link:
                atom1_name = self.get_charmm_name(atom1_index, top_dict, charmm_name_atom_struct, res_name_list)
                for atom2_index in atom3_link:
                    if atom2_index in [atom1_index]:
                        continue
                    atom2_name = self.get_charmm_name(atom2_index, top_dict, charmm_name_atom_struct, res_name_list)
                    for atom4_index in atom3_link:
                        if atom4_index in [atom1_index, atom2_index]:
                            continue
                        atom4_name = self.get_charmm_name(atom4_index, top_dict, charmm_name_atom_struct, res_name_list)

                        # if atom1_name == atom2_name and atom2_name == atom4_name and atom1_name == 'NC2':
                        #    print(atom1_index, atom2_index, atom4_index)

                        improper_key = '{} {} {} {}'.format(atom3_name, atom1_name, atom2_name, atom4_name)
                        improper_key_reverse = '{} {} {} {}'.format(atom1_name, atom2_name, atom4_name, atom3_name)
                        improper_X_key_4 = '{} X X {}'.format(atom3_name, atom4_name)
                        improper_X_key_reverse_4 = '{} X X {}'.format(atom4_name, atom3_name)
                        improper_X_key_2 = '{} X X {}'.format(atom3_name, atom2_name)
                        improper_X_key_reverse_2 = '{} X X {}'.format(atom2_name, atom3_name)
                        improper_X_key_1 = '{} X X {}'.format(atom3_name, atom1_name)
                        improper_X_key_reverse_1 = '{} X X {}'.format(atom1_name, atom3_name)

                        X_mat_key_4 = '{} {}'.format(atom3_index, atom4_index)
                        X_mat_key_1 = '{} {}'.format(atom3_index, atom1_index)
                        X_mat_key_2 = '{} {}'.format(atom3_index, atom2_index)

                        if improper_key in params_dict['IMPROPER']:
                            improper_type = params_dict['IMPROPER'][improper_key]
                            if atom3_index not in improper_pos_noX_mat:
                                improper_pos_noX_mat[atom3_index] = [{atom1_index, atom2_index, atom4_index}]
                                improper_mat.append(
                                    [[atom3_index, atom1_index, atom2_index, atom4_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom1_index, atom4_index, atom2_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom2_index, atom4_index, atom1_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom2_index, atom1_index, atom4_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom4_index, atom1_index, atom2_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom4_index, atom2_index, atom1_index], improper_type])

                            elif {atom1_index, atom2_index, atom4_index} not in improper_pos_noX_mat[atom3_index]:
                                improper_pos_noX_mat[atom3_index].append({atom1_index, atom2_index, atom4_index})
                                improper_mat.append(
                                    [[atom3_index, atom1_index, atom2_index, atom4_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom1_index, atom4_index, atom2_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom2_index, atom4_index, atom1_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom2_index, atom1_index, atom4_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom4_index, atom1_index, atom2_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom4_index, atom2_index, atom1_index], improper_type])

                        elif improper_key_reverse in params_dict['IMPROPER']:
                            improper_type = params_dict['IMPROPER'][improper_key_reverse]
                            if atom3_index not in improper_pos_noX_mat:
                                improper_pos_noX_mat[atom3_index] = [{atom1_index, atom2_index, atom4_index}]
                                improper_mat.append(
                                    [[atom3_index, atom1_index, atom2_index, atom4_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom1_index, atom4_index, atom2_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom2_index, atom4_index, atom1_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom2_index, atom1_index, atom4_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom4_index, atom1_index, atom2_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom4_index, atom2_index, atom1_index], improper_type])

                            elif {atom1_index, atom2_index, atom4_index} not in improper_pos_noX_mat[atom3_index]:
                                improper_pos_noX_mat[atom3_index].append({atom1_index, atom2_index, atom4_index})
                                improper_mat.append(
                                    [[atom3_index, atom1_index, atom2_index, atom4_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom1_index, atom4_index, atom2_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom2_index, atom4_index, atom1_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom2_index, atom1_index, atom4_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom4_index, atom1_index, atom2_index], improper_type])
                                improper_mat.append(
                                    [[atom3_index, atom4_index, atom2_index, atom1_index], improper_type])

                        elif improper_X_key_4 in params_dict['IMPROPER']:
                            improper_type = params_dict['IMPROPER'][improper_X_key_4]
                            if X_mat_key_4 not in improper_pos_X_mat:
                                improper_pos_X_mat[X_mat_key_4] = [{atom1_index, atom2_index}]
                                improper_mat_X.append(
                                    [[atom3_index, atom1_index, atom2_index, atom4_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom2_index, atom1_index, atom4_index], improper_type])
                            elif {atom1_index, atom2_index} not in improper_pos_X_mat[X_mat_key_4]:
                                improper_pos_X_mat[X_mat_key_4].append({atom1_index, atom2_index})
                                improper_mat_X.append(
                                    [[atom3_index, atom1_index, atom2_index, atom4_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom2_index, atom1_index, atom4_index], improper_type])
                        elif improper_X_key_reverse_4 in params_dict['IMPROPER']:
                            improper_type = params_dict['IMPROPER'][improper_X_key_reverse_4]
                            if X_mat_key_4 not in improper_pos_X_mat:
                                improper_pos_X_mat[X_mat_key_4] = [{atom1_index, atom2_index}]
                                improper_mat_X.append(
                                    [[atom3_index, atom1_index, atom2_index, atom4_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom2_index, atom1_index, atom4_index], improper_type])
                                improper_mat_X.append(
                                    [[atom4_index, atom1_index, atom2_index, atom3_index], improper_type])
                                improper_mat_X.append(
                                    [[atom4_index, atom2_index, atom1_index, atom3_index], improper_type])
                            elif {atom1_index, atom2_index} not in improper_pos_X_mat[X_mat_key_4]:
                                improper_pos_X_mat[X_mat_key_4].append({atom1_index, atom2_index})
                                improper_mat_X.append(
                                    [[atom3_index, atom1_index, atom2_index, atom4_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom2_index, atom1_index, atom4_index], improper_type])
                                improper_mat_X.append(
                                    [[atom4_index, atom1_index, atom2_index, atom3_index], improper_type])
                                improper_mat_X.append(
                                    [[atom4_index, atom2_index, atom1_index, atom3_index], improper_type])

                        elif improper_X_key_2 in params_dict['IMPROPER']:
                            improper_type = params_dict['IMPROPER'][improper_X_key_2]
                            if X_mat_key_2 not in improper_pos_X_mat:
                                improper_pos_X_mat[X_mat_key_2] = [{atom1_index, atom4_index}]
                                improper_mat_X.append(
                                    [[atom3_index, atom1_index, atom4_index, atom2_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom4_index, atom1_index, atom2_index], improper_type])
                            elif {atom1_index, atom4_index} not in improper_pos_X_mat[X_mat_key_2]:
                                improper_pos_X_mat[X_mat_key_2].append({atom1_index, atom4_index})
                                improper_mat_X.append(
                                    [[atom3_index, atom1_index, atom4_index, atom2_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom4_index, atom1_index, atom2_index], improper_type])
                        elif improper_X_key_reverse_2 in params_dict['IMPROPER']:
                            improper_type = params_dict['IMPROPER'][improper_X_key_reverse_2]
                            if X_mat_key_2 not in improper_pos_X_mat:
                                improper_pos_X_mat[X_mat_key_2] = [{atom1_index, atom4_index}]
                                improper_mat_X.append(
                                    [[atom3_index, atom1_index, atom4_index, atom2_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom4_index, atom1_index, atom2_index], improper_type])
                                improper_mat_X.append(
                                    [[atom2_index, atom1_index, atom4_index, atom3_index], improper_type])
                                improper_mat_X.append(
                                    [[atom2_index, atom4_index, atom1_index, atom3_index], improper_type])
                            elif {atom1_index, atom4_index} not in improper_pos_X_mat[X_mat_key_2]:
                                improper_pos_X_mat[X_mat_key_2].append({atom1_index, atom4_index})
                                improper_mat_X.append(
                                    [[atom3_index, atom1_index, atom4_index, atom2_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom4_index, atom1_index, atom2_index], improper_type])
                                improper_mat_X.append(
                                    [[atom2_index, atom1_index, atom4_index, atom3_index], improper_type])
                                improper_mat_X.append(
                                    [[atom2_index, atom4_index, atom1_index, atom3_index], improper_type])

                        elif improper_X_key_1 in params_dict['IMPROPER']:
                            improper_type = params_dict['IMPROPER'][improper_X_key_1]
                            if X_mat_key_1 not in improper_pos_X_mat:
                                improper_pos_X_mat[X_mat_key_1] = [{atom4_index, atom2_index}]
                                improper_mat_X.append(
                                    [[atom3_index, atom4_index, atom2_index, atom1_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom2_index, atom4_index, atom1_index], improper_type])

                            elif {atom4_index, atom2_index} not in improper_pos_X_mat[X_mat_key_1]:
                                improper_pos_X_mat[X_mat_key_1].append({atom4_index, atom2_index})
                                improper_mat_X.append(
                                    [[atom3_index, atom4_index, atom2_index, atom1_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom2_index, atom4_index, atom1_index], improper_type])

                        elif improper_X_key_reverse_1 in params_dict['IMPROPER']:
                            improper_type = params_dict['IMPROPER'][improper_X_key_reverse_1]
                            if X_mat_key_1 not in improper_pos_X_mat:
                                improper_pos_X_mat[X_mat_key_1] = [{atom4_index, atom2_index}]
                                improper_mat_X.append(
                                    [[atom3_index, atom4_index, atom2_index, atom1_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom2_index, atom4_index, atom1_index], improper_type])
                                improper_mat_X.append(
                                    [[atom1_index, atom4_index, atom2_index, atom3_index], improper_type])
                                improper_mat_X.append(
                                    [[atom1_index, atom2_index, atom4_index, atom3_index], improper_type])
                            elif {atom4_index, atom2_index} not in improper_pos_X_mat[X_mat_key_1]:
                                improper_pos_X_mat[X_mat_key_1].append({atom4_index, atom2_index})
                                improper_mat_X.append(
                                    [[atom3_index, atom4_index, atom2_index, atom1_index], improper_type])
                                improper_mat_X.append(
                                    [[atom3_index, atom2_index, atom4_index, atom1_index], improper_type])
                                improper_mat_X.append(
                                    [[atom1_index, atom4_index, atom2_index, atom3_index], improper_type])
                                improper_mat_X.append(
                                    [[atom1_index, atom2_index, atom4_index, atom3_index], improper_type])
        dihedral_mat = self.get_unique_list(dihedral_mat)

        # improper_mat = get_unique_list(improper_mat)
        def find_improper_mat(atoms):
            find_flag = False
            for x in improper_mat_:
                if [x[0][0], {x[0][1], x[0][2], x[0][3]}] == [atoms[0], {atoms[1], atoms[2], atoms[3]}]:
                    find_flag = True
                    break
            return find_flag

        improper_mat_ = []
        for x in improper_mat:
            atoms = x[0]
            atom1_name, atom2_name = charmm_name_atom_struct[atoms[0]], charmm_name_atom_struct[atoms[1]]
            atom3_name, atom4_name = charmm_name_atom_struct[atoms[2]], charmm_name_atom_struct[atoms[3]]
            atoms_name = [atom1_name, atom2_name, atom3_name, atom4_name]
            atom1_res_index, atom2_res_index = res_index_list[atoms[0]], res_index_list[atoms[1]]
            atom3_res_index, atom4_res_index = res_index_list[atoms[2]], res_index_list[atoms[3]]

            res_index_min = int(np.argmin([atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index]))
            res_min = min(atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index)
            min_flag = [atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index].count(res_min) == 1
            res_index_max = int(np.argmax([atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index]))
            res_max = max(atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index)
            max_flag = [atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index].count(res_max) == 1
            if max_flag:
                res_name = self.pdb.residues[res_max - 1].name
            elif min_flag:
                res_name = self.pdb.residues[res_min + 1].name
            else:
                res_name = self.pdb.residues[res_min].name

            if res_min == res_max:
                # res_name = res_name_list[res_min]
                if res_min == len(self.pdb.residues) - 1:
                    src_improper_list = top_dict['impr'][res_name].copy()
                    src_improper_list.append(['CC', 'OC', 'CA', 'OC'])
                else:
                    src_improper_list = top_dict['impr'][res_name]
                src_improper_list_ = [[z[0], {z[1], z[2], z[3]}] for z in src_improper_list]
                if [atoms_name[0], atoms_name[1], atoms_name[2], atoms_name[3]] in src_improper_list_:
                    if not find_improper_mat(atoms):
                        improper_mat_.append(x)
                #    print(res_name, res_min, atoms_name, src_improper_list)
                # else:
                #    print(res_name, res_min, atoms_name, src_improper_list)
            else:

                if min_flag and atoms_name[res_index_min] == 'C':
                    # res_name = res_name_list[res_min + 1]
                    atoms_name[res_index_min] = '-C'
                if max_flag and atoms_name[res_index_max] == 'N':
                    # res_name = res_name_list[res_max - 1]
                    atoms_name[res_index_max] = '+N'
                src_improper_list = top_dict['impr'][res_name]
                src_improper_list_ = [[z[0], {z[1], z[2], z[3]}] for z in src_improper_list]
                if [atoms_name[0], atoms_name[1], atoms_name[2], atoms_name[3]] in src_improper_list_:
                    if not find_improper_mat(atoms):
                        improper_mat_.append(x)
                #   print(res_name, res_max, atoms_name, src_improper_list)
                # else:
                #    print(res_name, res_min, atoms_name, src_improper_list)

        for x in improper_mat_X:
            atoms = x[0]
            improper_type = x[1]
            atoms_reverse = atoms.copy()
            atoms_reverse[1], atoms_reverse[2] = atoms_reverse[2], atoms_reverse[1]
            atom1_name, atom2_name = charmm_name_atom_struct[atoms[0]], charmm_name_atom_struct[atoms[1]]
            atom3_name, atom4_name = charmm_name_atom_struct[atoms[2]], charmm_name_atom_struct[atoms[3]]
            atoms_name = [atom1_name, atom2_name, atom3_name, atom4_name]
            atom1_res_index, atom2_res_index = res_index_list[atoms[0]], res_index_list[atoms[1]]
            atom3_res_index, atom4_res_index = res_index_list[atoms[2]], res_index_list[atoms[3]]

            res_index_min = int(np.argmin([atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index]))
            res_min = min(atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index)
            min_flag = [atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index].count(res_min) == 1
            res_index_max = int(np.argmax([atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index]))
            res_max = max(atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index)
            max_flag = [atom1_res_index, atom2_res_index, atom3_res_index, atom4_res_index].count(res_max) == 1
            if max_flag:
                res_name = self.pdb.residues[res_max - 1].name
            elif min_flag:
                res_name = self.pdb.residues[res_min + 1].name
            else:
                res_name = self.pdb.residues[res_min].name

            if res_min == res_max:
                # res_name = res_name_list[res_min]
                if res_min == len(self.pdb.residues) - 1:
                    src_improper_list = top_dict['impr'][res_name].copy()
                    src_improper_list.append(['CC', 'OC', 'CA', 'OC'])
                else:
                    src_improper_list = top_dict['impr'][res_name]
                src_improper_list_ = [[z[0], {z[1], z[2]}, z[3]] for z in src_improper_list]
                if [atoms_name[0], {atoms_name[1], atoms_name[2]}, atoms_name[3]] in src_improper_list_:
                    if not find_improper_mat(atoms):
                        improper_mat_.append(x)
                #    print(res_name, res_min, atoms_name, src_improper_list)
                # else:
                #    print(res_name, res_min, atoms_name, src_improper_list)
            else:

                if min_flag and atoms_name[res_index_min] == 'C':
                    # res_name = res_name_list[res_min + 1]
                    atoms_name[res_index_min] = '-C'
                if max_flag and atoms_name[res_index_max] == 'N':
                    # res_name = res_name_list[res_max - 1]
                    atoms_name[res_index_max] = '+N'
                src_improper_list = top_dict['impr'][res_name]
                src_improper_list_ = [[z[0], {z[1], z[2]}, z[3]] for z in src_improper_list]
                if [atoms_name[0], {atoms_name[1], atoms_name[2]}, atoms_name[3]] in src_improper_list_:
                    # if not find_improper_mat(atoms):
                    improper_mat_.append(x)
                # else:
                #    print(res_name, res_min, atoms_name, src_improper_list)

        # improper_mat =  get_unique_list(improper_mat_)
        improper_mat = improper_mat_

        return dihedral_mat, improper_mat

    def get_unique_list(self, src_list):
        unique_list = []
        for x in src_list:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list
