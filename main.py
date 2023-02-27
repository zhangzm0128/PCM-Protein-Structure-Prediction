import os
import json
import argparse
from utils.LogUtils import *
from search.PCM import *
from utils.ProteinUtils import *
import time
from energy.energy_nofile import *

parser = argparse.ArgumentParser()


parser.add_argument('--config', type=str, default='config/config_knea.json',
                    help='the path of global config file.')
parser.add_argument('--protein_config', type=str, default='protein_config.json',
                    help='the path of protein config file.')
parser.add_argument('--energy_config', type=str, default='config/energy_config.json',
                    help='the path of energy function config file.')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='the path of checkpoint and program will run checkpoint data.')
parser.add_argument('--mode', type=str, default=None)

args = parser.parse_args()
checkpoint = args.checkpoint
# set Logger and init logging save path
logging = LoggerWriter(args)
logs_root, energy_temp_save_path, protein_save_path, config_save = logging.init_path()

# read configuration file
config_file = open(os.path.join(config_save, 'config.json'), 'r').read()
config = json.loads(config_file)

protein_config_file = open(os.path.join(config_save, 'protein_config.json'), 'r').read()
protein_config = json.loads(protein_config_file)

energy_config_file = open(os.path.join(config_save, 'energy_config.json'), 'r').read()
energy_config = json.loads(energy_config_file)

# init parameters
second_struct_file_path = config['protein_params']['second_struct_file']
protein_status = config['protein_params']['status']
protein_name = config['protein_params']['name']
root = config['paths']['root']

pop_size = config['algo_params']['pop_size']

num_obj = config['energy_params']['number_objective']
max_thread = config['energy_params']['max_thread']

# set coder and energy
coder = Coding(protein_config, protein_status)


proteins = []
# init protein
if checkpoint == None:
    # generate proteins
    for x in range(pop_size):
        new_protein = Protein(num_obj, protein_status, coder)
        proteins.append(coder.decoder_from_seq(second_struct_file_path, new_protein))

    # calculate energy of init proteins
    start_time = time.time()
    #energy.calculate_energy(proteins)
    #logging.write(proteins, coder, config['algo_params']['save_all'], 0)
else:
    for x in range(pop_size):
        data_path = os.path.join(protein_save_path, protein_name + '_' + str(x))
        new_protein = Protein(num_obj, protein_status, coder)
        proteins.append(coder.decoder_from_logger(data_path, new_protein))
energy = Energy(energy_config, root, energy_temp_save_path, protein_name, second_struct_file_path, max_thread, proteins[0])
energy.calculate_energy(proteins)
logging.write(proteins, coder, config['algo_params']['save_all'], 0)

end_time = time.time()
print('total time', end_time - start_time)
search_algo = PCM(proteins, config['algo_params'], energy, coder, logging, current_gen=1)

search_algo.run()
