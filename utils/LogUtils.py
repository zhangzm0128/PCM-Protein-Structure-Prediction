import os
import time
import json
from utils.ProteinUtils import *
from shutil import copyfile


class LoggerWriter():
    def __init__(self, input_args, use_ref=False):
        self.input_args = input_args
        self.use_ref = use_ref

    def init_path(self):
        if self.input_args.checkpoint == None:
            config_file = open(self.input_args.config, 'r').read()
            config = json.loads(config_file)

            self.protein_name = config['protein_params']['name']

            logs_root = config['paths']['logs_root']
            create_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
            self.logs_root = os.path.join(logs_root, create_time + '_' + self.protein_name)
            os.mkdir(self.logs_root)

            self.energy_temp_save_path = os.path.join(self.logs_root, 'energy_temp_save')
            self.protein_save_path = os.path.join(self.logs_root, 'protein_save')
            self.config_save = os.path.join(self.logs_root, 'config_save')

            os.mkdir(self.energy_temp_save_path)
            os.mkdir(self.protein_save_path)
            os.mkdir(self.config_save)

            copyfile(self.input_args.config, os.path.join(self.config_save, 'config.json'))
            copyfile(self.input_args.protein_config, os.path.join(self.config_save, 'protein_config.json'))
            copyfile(self.input_args.energy_config, os.path.join(self.config_save, 'energy_config.json'))

            if self.use_ref == False:
                self.num_proteins = config['algo_params']['pop_size']
            else:
                self.num_proteins = config['algo_params']['reference_size']
            for x in range(self.num_proteins):
                protein_solution_folder = os.path.join(self.protein_save_path, self.protein_name + '_' + str(x))
                os.mkdir(protein_solution_folder)

        else:
            self.logs_root = self.input_args.checkpoint
            self.energy_temp_save_path = os.path.join(self.logs_root, 'energy_temp_save')
            self.protein_save_path = os.path.join(self.logs_root, 'protein_save')
            self.config_save = os.path.join(self.logs_root, 'config_save')

        return self.logs_root, self.energy_temp_save_path, self.protein_save_path, self.config_save



    def creat(self):
        self.file_name = os.path.join(self.path, time.strftime('%Y-%m-%d,%H-%M-%S', time.localtime(time.time())) + '-log.csv')
        with open(os.path.join(self.file_name), 'w') as f:
            f.write(','.join(self.header) + '\n')

    def write(self, proteins, coder, save_all, current_gen):
        for protein_index, protein in enumerate(proteins):
            if protein_index >= self.num_proteins:
                break
            data_path = os.path.join(self.protein_save_path, self.protein_name + '_' + str(protein_index))
            coder.encoder_to_logger(data_path, protein, save_all, current_gen)

    def write_pdb(self, current_gen):
        pdb_save_root = os.path.join(self.energy_temp_save_path, 'xyz_pdb')
        for pdb_file in os.listdir(pdb_save_root):
            protein_index = pdb_file.split('.pdb')[0].split('_')[-1]
            if protein_index == 'demo':
                continue
            data_path = os.path.join(self.protein_save_path, self.protein_name + '_' + str(protein_index))
            copyfile(os.path.join(pdb_save_root, pdb_file), os.path.join(data_path, str(current_gen) + '.pdb'))

    def write_archive(self, proteins, coder):
        for protein_index, protein in enumerate(proteins):
            data_path = os.path.join(os.path.dirname(self.protein_save_path), 'archive_protein')
            coder.encoder_to_loggerforarchive(data_path, protein, protein_index)
            
    def add_protein_save_path(self, new_num_proteins):
        if new_num_proteins > self.num_proteins:
            for x in range(self.num_proteins, new_num_proteins):
                protein_solution_folder = os.path.join(self.protein_save_path, self.protein_name + '_' + str(x))
                os.mkdir(protein_solution_folder)
            self.num_proteins = new_num_proteins







