import os
import shutil
from multiprocessing import Process, Queue, current_process, freeze_support, cpu_count
from threading import Thread, Semaphore
import subprocess
import time
import attr
import re
import pyrosetta
from energy.Charmm import Charmm

import pyrosetta
from pyrosetta.rosetta import core
from pyrosetta.rosetta.core.scoring.sasa import SasaCalc


from utils.ProteinUtils import *


class Energy(object):
    def __init__(self, config, root, log_root, protein_name, second_struct_path, max_thread, pdb_demo):
        pyrosetta.init('-out:level 0')
        self.config = config
        self.root = root
        self.log_root = log_root
        self.protein_name = protein_name
        self.second_struct_path = second_struct_path
        self.max_thread = max_thread

        self.energy_init()
        self.read_structure()
        self.init_save_dir()
        self.init_general_params(max_thread)
        self.init_Rosetta_params()
        self.init_Charmm(pdb_demo)
        self.init_non_bond()
        
        self.task_queue = Queue()
        self.done_queue = Queue()
        
        self.first_create = True

        self.save_all = self.config['general_params']['save_all']
        self.update_prefix_name()
        
        self.use_core = cpu_count() - 2


    def init_general_params(self, max_thread):
        self.prm_path = self.config['general_params']['prm_path']
        #self.pool = Pool(max_thread)
        self.max_thread = Semaphore(max_thread)

    def init_Rosetta_params(self):

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

    def init_non_bond(self):
        self.non_bond_fxn = pyrosetta.ScoreFunction()
        self.non_bond_fxn.set_weight(core.scoring.fa_atr, 1.0)
        self.non_bond_fxn.set_weight(core.scoring.fa_rep, 1.0)
        self.non_bond_fxn.set_weight(core.scoring.fa_elec, 1.0)

    def init_Charmm(self, pdb_demo):
        parameters_file_path = self.config['general_params']['charmm_prm']
        topology_file_path = self.config['general_params']['charmm_top']
        pose = pyrosetta.pose_from_sequence(self.protein_seq, "fa_standard")
        for res_index, x in enumerate(pdb_demo.res):
            phi = x.get_angle('phi')
            psi = x.get_angle('psi')
            omega = x.get_angle('omega')
            sidechain = x.get_angle('sidechain')
            pose.set_phi(res_index + 1, phi)
            pose.set_psi(res_index + 1, psi)
            pose.set_omega(res_index + 1, omega)
            for chi_index, chi in enumerate(sidechain):
                pose.set_chi(chi_index + 1, res_index + 1, chi)
        save_path = os.path.join(self.xyz_pdb_root, 'demo' + '.pdb')
        pose.dump_pdb(save_path)
        self.charmm = Charmm(parameters_file_path, topology_file_path, save_path)
    
    
    def bond_wrapper(self, objective_dict, protein):
        bond_energy = self.charmm.calculate_unit(protein.protein_path)
        return bond_energy

    def Rosetta_wrapper(self, objective_dict, protein):
        pose = pyrosetta.pose_from_file(protein.protein_path)
        score = self.scorefxn(pose)
        return score 

    def non_bond_wrapper(self, objective_dict, protein):
        pose = pyrosetta.pose_from_file(protein.protein_path)
        score = self.non_bond_fxn(pose)
        return score 

    def RWplus_wrapper(self, objective_dict, protein):
        bin = objective_dict['bin_path']

        # pdb_save_path = os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index) + '.pdb')
        # pdb_save_path = self.add_root(pdb_save_path)
        pdb_save_path = protein.protein_path
        command = '%s %s' % (bin, pdb_save_path)
        bin_out = self.external_command(command)

        line = bin_out[0].decode()
        numbers = line.split()
        RWplus_energy = float(numbers[3])
        
        return RWplus_energy
    
    def dDFIRE_wrapper(self, objective_dict, protein):
        bin = objective_dict['bin_path']
        export_path = objective_dict['param_path']

        # pdb_save_path = os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index) + '.pdb')
        # pdb_save_path = self.add_root(pdb_save_path)
        pdb_save_path = protein.protein_path
        command = 'export DATADIR="{}"; {} {}'.format(export_path, bin, pdb_save_path)
        bin_out = self.external_command(command)

        line = bin_out[0].decode()
        numbers = line.split()
        dDFIRE_energy = float(numbers[1])
        
        return dDFIRE_energy
        
    def SASA_wrapper(self, objective_dict, protein):
        pdb_save_path = protein.protein_path
        pose = pyrosetta.pose_from_pdb(pdb_save_path)
        sasacalc = SasaCalc()
        sasacalc.calculate(pose)
        sasa = sasacalc.get_total_sasa()
        return sasa
        
    def CHARMM_wrapper(self, objective_dict, protein):
        pdb_save_path = protein.protein_path
        generate_xyz_bin = objective_dict['generate_xyz']
        param_file = objective_dict['param_path']
        generate_xyz_command = '{} {} {}'.format(generate_xyz_bin, pdb_save_path, param_file)
        self.external_command(generate_xyz_command)
        xyz_save_path = pdb_save_path.replace('.pdb', '.xyz')
        
        charmm_bin = objective_dict['charmm_bin']
        charmm_command = '{} {} {} E'.format(charmm_bin, xyz_save_path, param_file)
        bin_out = self.external_command(charmm_command)
        '''
        The output format of Tinker (line_id starts from 0)
        line_id  details
        ...      ...
        14       b'\n'
        15       b' Total Potential Energy :              0.8575D+11 Kcal/mole\n'
        16       b'\n'
        17       b' Energy Component Breakdown :           Kcal/mole        Interactions\n'
        18       b'\n'
        19       b' Bond Stretching                         154.4335              933\n'
        20       b' Angle Bending                            98.8979             1694\n'
        21       b' Urey-Bradley                             37.9606              836\n'
        22       b' Improper Dihedral                         0.0024              180\n'
        23       b' Torsional Angle                         326.7167             2487\n'
        24       b' Van der Waals                         0.8575D+11           425648\n'
        25       b' Charge-Charge                          -124.0853           419261\n'
        '''
        def format_out(out):
            return float(out.decode('ascii').replace('D', 'E'))
            
        bond = format_out(bin_out[19].split()[2])
        angle = format_out(bin_out[20].split()[2])
        ub = format_out(bin_out[21].split()[1])
        improper = format_out(bin_out[22].split()[2])
        torsion = format_out(bin_out[23].split()[2])
        bonded = bond + angle + ub + improper + torsion
        
        lj = format_out(bin_out[24].split()[3])
        charge = format_out(bin_out[25].split()[1])
        non_bonded = lj + charge
        
        CHARMM_mode = objective_dict['mode']
        if CHARMM_mode == 'all':
            return [bonded, non_bonded]
        elif CHARMM_mode == 'bond':
            return bonded
        elif CHARMM_mode == 'non_bond':
            return non_bonded
        
   
        
        

    def calculate_energy_async(self, task_queue, done_queue):
        def thread_function_calc(protein, protein_index, objective_dict, done_queue, energy_wrapper):
            with max_thread:
                objective_index = objective_dict['objective_index']
                score = energy_wrapper(objective_dict, protein)
                done_queue.put([score, protein_index, objective_index])
                
                
        def thread_function_generate(protein, protein_index, done_queue):
            with max_thread:
                pdb_save_path = self.generate_pdb_rosetta(protein_index, protein)
                done_queue.put([protein_index, pdb_save_path])
        
        max_thread = Semaphore(32)
        thread_id = []
        for energy_name, protein_detail in iter(task_queue.get, 'STOP'):
            if energy_name == 'generate':
                protein_index, protein = protein_detail
                generate_thread = Thread(target=thread_function_generate, args=(protein, protein_index, done_queue))
                generate_thread.start()
                thread_id.append(generate_thread)
            else:                
                objective_dict = self.energy_dict[energy_name]
                energy_wrapper = getattr(self, energy_name + '_wrapper')
                protein_index, protein = protein_detail
                cal_energy_thread = Thread(target=thread_function_calc, args=(protein, protein_index, objective_dict, done_queue, energy_wrapper))
                cal_energy_thread.start()
                thread_id.append(cal_energy_thread)
        for x in thread_id:
            x.join()
    
    def calculate_energy(self, proteins):
        self.put_generate_task(proteins)
        self.get_generate_result(proteins)
        
        self.put_energies_task(proteins)
        self.get_energies_result(proteins)
        
        
    def stop(self):
        for _ in range(self.use_core):
            self.task_queue.put('STOP')
            
    def put_generate_task(self, proteins):
        for protein_index, protein in enumerate(proteins):
            self.task_queue.put(['generate', (protein_index, protein)])
        if self.first_create:
            self.create_calculate_process()
            self.first_create = False
        
    def put_energies_task(self, proteins):
        for energy_name in self.energy_dict:
            for protein_index, protein in enumerate(proteins):
                self.task_queue.put([energy_name, (protein_index, protein)])
                
    def get_energies_result(self, proteins):
        for _ in range(len(self.energy_dict) * len(proteins)):
            score, protein_index, objective_index = self.done_queue.get()
            objective_index = np.array(objective_index)
            proteins[protein_index].obj[objective_index] = score
            
    def get_generate_result(self, proteins):
        for _ in range(len(proteins)):
            protein_index, protein_path = self.done_queue.get()
            proteins[protein_index].protein_path = protein_path
            
            
    def create_calculate_process(self):
        for x in range(self.use_core):
            Process(target=self.calculate_energy_async, args=(self.task_queue, self.done_queue)).start()
           

    def generate_pdb_rosetta(self, protein_index, protein):
    
        pose = pyrosetta.pose_from_sequence(self.protein_seq, "fa_standard")
        for res_index, x in enumerate(protein.res):
            phi = x.get_angle('phi')
            psi = x.get_angle('psi')
            omega = x.get_angle('omega')
            sidechain = x.get_angle('sidechain')
            pose.set_phi(res_index + 1, phi)
            pose.set_psi(res_index + 1, psi)
            pose.set_omega(res_index + 1, omega)
            for chi_index, chi in enumerate(sidechain):
                pose.set_chi(chi_index + 1, res_index + 1, chi)
        protein.set_rosetta_pose(pose)
        pose.dump_pdb(os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index) + '.pdb'))
        
        pdb_save_path = os.path.join(self.xyz_pdb_root, self.prefix_name + '_' + str(protein_index) + '.pdb')
        
        return pdb_save_path

    def read_structure(self):
        structure_file = open(self.second_struct_path, 'r')
        res_full_name_dict = self.config['protein_shorthand']
        self.protein_seq = ''
        for x in structure_file:
            res_full_name = x.split(' ')[0]
            self.protein_seq = self.protein_seq + res_full_name_dict[res_full_name]




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

        out = process.stdout.readlines()
        return out

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
