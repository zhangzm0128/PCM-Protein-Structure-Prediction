import os
import shutil
from threading import Thread, Semaphore
import subprocess
import time
import attr
import re
import pyrosetta

from pyrosetta.rosetta import core
from utils.ProteinUtils import *

class Energy(object):
    def __init__(self, config, root, log_root, protein_name, second_struct_path, max_thread):
        self.config = config
        self.root = root
        self.log_root = log_root
        self.protein_name = protein_name
        self.second_struct_path = second_struct_path
        self.max_thread = max_thread

        self.energy_init()
        self.init_save_dir()
        self.init_general_params()
        self.init_Rosetta_params()

        self.save_all = self.config['general_params']['save_all']
        self.update_prefix_name()

    def init_general_params(self):
        self.prm_path = self.config['general_params']['prm_path']
        self.max_thread = Semaphore(self.max_thread)

    def init_Rosetta_params(self):
        pyrosetta.init('-out:level 0')

        self.scorefxn = pyrosetta.ScoreFunction()
        self.scorefxn.set_weight(core.scoring.fa_atr, 0.800)  # full-atom attractive score
        self.scorefxn.set_weight(core.scoring.fa_rep, 0.440)  # full-atom repulsive score
        self.scorefxn.set_weight(core.scoring.fa_sol, 0.750)  # full-atom solvation score
        self.scorefxn.set_weight(core.scoring.fa_intra_rep, 0.004)  # f.a. intraresidue rep. score
        self.scorefxn.set_weight(core.scoring.fa_elec, 0.700)  # full-atom electronic score
        self.scorefxn.set_weight(core.scoring.pro_close, 1.000)  # proline closure
        self.scorefxn.set_weight(core.scoring.hbond_sr_bb, 1.170)  # short-range hbonding
        self.scorefxn.set_weight(core.scoring.hbond_lr_bb, 1.170)  # long-range hbonding
        self.scorefxn.set_weight(core.scoring.hbond_bb_sc, 1.170)  # backbone-sidechain hbonding
        self.scorefxn.set_weight(core.scoring.hbond_sc, 1.100)  # sidechain-sidechain hbonding
        self.scorefxn.set_weight(core.scoring.dslf_fa13, 1.000)  # disulfide full-atom score
        self.scorefxn.set_weight(core.scoring.rama, 0.200)  # ramachandran score
        self.scorefxn.set_weight(core.scoring.omega, 0.500)  # omega torsion score
        self.scorefxn.set_weight(core.scoring.fa_dun, 0.560)  # fullatom Dunbrack rotamer score
        self.scorefxn.set_weight(core.scoring.p_aa_pp, 0.320)
        self.scorefxn.set_weight(core.scoring.ref, 1.000)  # reference identity score




    def energy_init(self):
        self.energy_dict = {}
        objective_index = 0
        objective_key = 'objective' + str(objective_index)
        while objective_key in self.config:
            energy_name = self.config[objective_key]['name']
            self.energy_dict[energy_name] = self.config[objective_key]
            del self.energy_dict[energy_name]['name']
            #self.energy_dict[energy_name]['objective_index'] = objective_index
            objective_index = objective_index + 1
            objective_key = 'objective' + str(objective_index)

    def add_root(self, path):
        return os.path.join(self.root, path)

    def external_command(self, command):
        process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        process.stdin.write('exit\n'.encode())
        while subprocess.Popen.poll(process) == None:
            continue

    def init_save_dir(self):
        self.calculate_protein_root = os.path.join(self.log_root, 'calculate_protein')
        os.mkdir(self.calculate_protein_root)

        self.xyz_pdb_root = os.path.join(self.log_root, 'xyz_pdb')
        os.mkdir(self.xyz_pdb_root)

        for x in self.energy_dict:
            energy_file = os.path.join(self.log_root, x)
            os.mkdir(energy_file)

    def update_prefix_name(self):
        self.prefix_name = self.config['general_params']['prefix']
        if self.prefix_name == "":
            self.prefix_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            self.prefix_name = self.prefix_name + '_' + self.protein_name

    def generate_calculate_protein_file(self, protein_index, protein):
        cal_protein_name = os.path.join(self.calculate_protein_root,
                                        self.prefix_name + '_' + str(protein_index))
        cal_protein_file = open(cal_protein_name + '.dat', 'w')

        xyz_save_path = os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index))

        # Write header information to the file
        cal_protein_file.write(self.add_root(xyz_save_path) + '\n') # set save xyz file path and save prefix
        cal_protein_file.write(self.add_root(self.second_struct_path) + '\n') # set second sturct file
        cal_protein_file.write(self.add_root(self.prm_path) + '\n') # set prm param file

        # Write angle to the file
        for x in protein.res:
            residue_name = x.name
            phi = x.get_angle('phi')
            psi = x.get_angle('psi')
            omega = x.get_angle('omega')
            sidechain = x.get_angle('sidechain')

            write_line = '%s %f %f %f' % (residue_name, phi, psi, omega)
            for y in sidechain:
                write_line = write_line + ' %f' % y

            cal_protein_file.write(write_line + '\n')

        cal_protein_file.write('\nn\n\n') # set Cyclize the Polypeptide Chain
        cal_protein_file.close()

    def calculate_energy_async(self, protein_index, protein):
        with self.max_thread:
            self.generate_calculate_protein_file(protein_index, protein)
            #self.generate_xyz_file_wrapper(protein_index)
            #self.generate_pdb_from_xyz_wrapper(protein_index)
            self.generate_xyz_pdb_file(protein_index)
            for x in self.energy_dict:
                energy_wrapper = getattr(self, x + '_wrapper')
                energy_save_path = energy_wrapper(x, self.energy_dict[x], protein_index)

                get_energy_wrapper = getattr(self, 'get_' + x)
                get_energy_wrapper(energy_save_path, protein, self.energy_dict[x]['objective_index'])



    def calculate_energy(self, proteins, proteins_indexs=None):
        thread_id = []
        if proteins_indexs is None:
            for protein_index, protein in enumerate(proteins):
                cal_energy_thread = Thread(target=self.calculate_energy_async, args=(protein_index, protein,))
                cal_energy_thread.start()
                thread_id.append(cal_energy_thread)
        else:
            for protein_index, protein in zip(proteins_indexs, proteins):
                cal_energy_thread = Thread(target=self.calculate_energy_async, args=(protein_index, protein,))
                cal_energy_thread.start()
                thread_id.append(cal_energy_thread)
        for x in thread_id:
            x.join()
        self.remove_temp_file()
        # for x in self.energy_dict:
        #     energy_wrapper = getattr(self, x + '_wrapper')
        #     energy_save_path = energy_wrapper(x, self.energy_dict[x], protein_index)
        #
        #     get_energy_wrapper = getattr(self, 'get_' + x)
        #     get_energy_wrapper(energy_save_path, protein, self.energy_dict[x]['objective_index'])


    def generate_xyz_pdb_file(self, protein_index):
        self.generate_xyz_file_wrapper(protein_index)
        self.generate_pdb_from_xyz_wrapper(protein_index)
        pdb_file_path = os.path.join(self.xyz_pdb_root,
                                     self.prefix_name + '_' + str(protein_index) + '.pdb')
        pdb_file_path = self.add_root(pdb_file_path)
        
        
        self.norm_pdb_file(pdb_file_path)

    def norm_pdb_file(self, pdb_file_path):
        src_file = open(pdb_file_path, 'r')
        
        src_lines = src_file.readlines()
        src_file.close()
        
        new_file = open(pdb_file_path, 'w')
        for line in src_lines[:3]:
            new_file.write(line)
        
        for line in src_lines[3:]:
            if len(line) > 54:
                temp = line[:30]
                num = re.findall(r'-?\d+\.?\d*', line)
                x = num[-3]
                x = x.rjust(8, ' ')
                y = num[-2]
                y = y.rjust(8, ' ')
                z = num[-1]
                z = z.rjust(8, ' ')
                temp = temp + x + y + z
                new_file.write(temp+'\n')
        new_file.close()

    def generate_xyz_file_wrapper(self, protein_index):
        bin = self.config['general_params']['generate_xyz_bin_path']
        bin = self.add_root(bin)
        cal_protein_name = os.path.join(self.calculate_protein_root,
                                        self.prefix_name + '_' + str(protein_index) + '.dat')
        cal_protein_name = self.add_root(cal_protein_name)
        command = '%s < %s ' % (bin, cal_protein_name)
        self.external_command(command)
        #subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def generate_pdb_from_xyz_wrapper(self, protein_index):
        bin = self.config['general_params']['generate_pdb_from_xyz_bin_path']
        bin = self.add_root(bin)
        xyz_save_path = os.path.join(self.xyz_pdb_root,
                                     self.prefix_name + '_' + str(protein_index) + '.xyz')
        xyz_save_path = self.add_root(xyz_save_path)
        pdb_save_path = os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index) + '.pdb')

        command = '%s %s %s' % (bin, xyz_save_path, self.add_root(self.prm_path))
        self.external_command(command)
        generation = int((len(os.listdir(os.path.join(self.log_root, '..', 'protein_save',
                                                 self.protein_name + '_' + str(protein_index)))) - 1) / 2)
        shutil.copy(pdb_save_path, os.path.join(self.log_root, '..', 'protein_save',
                                                self.protein_name + '_' + str(protein_index), 'solution_{}.pdb'.format(generation)))
        #subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # energy function wrapper
    def analyze_wrapper(self, name, objective_dict, protein_index):
        bin = objective_dict['bin_path']
        #bin = self.add_root(bin)
        xyz_save_path = os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index) + '.xyz')
        energy_save_path = os.path.join(self.log_root, name, 'energy' + '_' + str(protein_index) + '.dat')
        xyz_save_path = self.add_root(xyz_save_path)
        #energy_save_path = self.add_root(energy_save_path)
        command = '%s %s %s E > %s' % (bin, xyz_save_path, self.add_root(self.prm_path), energy_save_path)
        #os.system(command)
        self.external_command(command)
        return energy_save_path

    def sas_wrapper(self, name, objective_dict, protein_index):
        bin = objective_dict['bin_path']
        #bin = self.add_root(bin)
        pdb_save_path = os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index) + '.pdb')
        sas_save_path = os.path.join(self.log_root, name, 'sas')
        sas_save_path = self.add_root(sas_save_path)

        energy_save_path = os.path.join(self.log_root, name, 'energy' + '_' + str(protein_index) + '.dat')
        pdb_save_path = self.add_root(pdb_save_path)
        #energy_save_path= self.add_root(energy_save_path)
        command = '%s -i %s -s 2 -o %s > %s' % (bin, pdb_save_path, sas_save_path, energy_save_path)
        #os.system(command)
        self.external_command(command)
        return energy_save_path

    def Rosetta_wrapper(self, name, objective_dict, protein_index):
        pdb_save_path = os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index) + '.pdb')
        pdb_save_path = self.add_root(pdb_save_path)
        pose = pyrosetta.Pose()
        pyrosetta.pose_from_file(pose, pdb_save_path)
        score = self.scorefxn(pose)
        return score

    def RWplus_wrapper(self, name, objective_dict, protein_index):
        bin = objective_dict['bin_path']

        pdb_save_path = os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index) + '.pdb')
        energy_save_path = os.path.join(self.log_root, name, 'energy' + '_' + str(protein_index) + '.dat')
        pdb_save_path = self.add_root(pdb_save_path)
        command = '%s %s > %s' % (bin, pdb_save_path, energy_save_path)
        self.external_command(command)
        return energy_save_path


    # read energy function file
    def get_analyze(self, energy_save_path, protein, objective_index):
        energy_file = open(energy_save_path, 'r')
        for i in range(15):
            energy_file.readline()
        total_energy = ''
        stretching = ''
        angle = ''
        urey = ''
        improper = ''
        van = ''
        torsional = ''
        charge = ''
        for line in energy_file.readlines():
            numbers = line.split()
            if len(numbers) != 0:
                if numbers[0] == "Total":
                    total_energy = numbers[4]
                    total_energy = total_energy.replace("D", "E")
                elif numbers[0] == "Bond":
                    stretching = numbers[2]
                    stretching = stretching.replace("D", "E")
                elif numbers[0] == "Angle":
                    angle = numbers[2]
                    angle = angle.replace("D", "E")
                elif numbers[0] == "Urey-Bradley":
                    urey = numbers[1]
                    urey = urey.replace("D", "E")
                elif numbers[0] == "Improper":
                    improper = numbers[2]
                    improper = improper.replace("D", "E")
                elif numbers[0] == "Torsional":
                    torsional = numbers[2]
                    torsional = torsional.replace("D", "E")
                elif numbers[0] == "Van":
                    van = numbers[3]
                    van = van.replace("D", "E")
                elif numbers[0] == "Charge-Charge":
                    charge = numbers[1]
                    charge = charge.replace("D", "E")
        bond = float(stretching) + float(angle) + float(urey) + float(improper) + float(torsional)
        non_bond = float(van) + float(charge)
        protein.obj[objective_index[0]] = bond
        protein.obj[objective_index[1]] = non_bond

    def get_Rosetta(self, score, protein, objective_index):
        protein.obj[objective_index] = score

    def get_sas(self, energy_save_path, protein, objective_index):
        energy_file = open(energy_save_path, 'r')
        sas_energy = 0
        for line in energy_file.readlines():
            numbers = line.split()
            if numbers[0] == "Total" and numbers[1] == "area":
                sas_energy = float(numbers[2])
        protein.obj[objective_index] = sas_energy

    def get_RWplus(self,energy_save_path, protein, objective_index):
        energy_file = open(energy_save_path, 'r')
        RWplus_energy = 0
        line = energy_file.readline()
        numbers = line.split()
        RWplus_energy = float(numbers[3])
        protein.obj[objective_index] = RWplus_energy


    def remove_temp_file(self):
        shutil.rmtree(self.calculate_protein_root)
        os.mkdir(self.calculate_protein_root)

        shutil.rmtree(self.xyz_pdb_root)
        os.mkdir(self.xyz_pdb_root)

        for x in self.energy_dict:
            energy_file = os.path.join(self.log_root, x)
            shutil.rmtree(energy_file)
            os.mkdir(energy_file)
