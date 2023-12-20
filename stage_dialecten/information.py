tmit_path = "/home/zjoske/Documents/tmit"
data_path = "/media/zjoske/Seagate Expansion Drive/scriptie"
output_path = "/home/zjoske/Documents/stage_outcome"

attempt = "four_layers_weighted"

phones = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix',
          'axr', 'ax-h', 'jh', 'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh',
          'm', 'n', 'ng', 'em', 'nx', 'en', 'eng', 'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl',
          'tcl', 'kcl', 'q', 'pau', 'epi', 'h#']
vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix',
          'axr', 'ax-h']

all_info = {'all_vowels': vowels,
            'all_i': ['ih', 'ix', 'iy'],
            'all_u': ['uh', 'uw', 'ux'],
            'all_a': ['aa', 'ae', 'ah'], 'all_mid': ['ax-h', 'ax', 'axr', 'eh', 'ay', 'er', 'ow'], 'just_i': ['iy'],
            'just_u': ['uw'], 'just_a': ['aa'], 'just_mid': ['ax']
            }

group_data = {'iy': 'all_i', 'ih': 'all_i', 'eh': 'all_mid', 'ey': 'all_vowels', 'ae': 'all_a', 'aa': 'all_a',
                     'aw': 'all_vowels', 'ay': 'all_mid', 'ah': 'all_a', 'ao': 'all_vowels', 'oy': 'all_vowels',
                     'ow': 'all_mid', 'uh': 'all_u', 'uw': 'all_u', 'ux': 'all_u', 'er': 'all_mid', 'ax': 'all_mid',
                     'ix': 'all_i', 'axr': 'all_mid', 'ax-h': 'all_mid'}
reduce_groups = False


dic_mfcc = {'n_mels': 13,  ##must be int
            'hoplength': 160,  ## must be int
            'framelength': 400}  ## must be int

dic_design = {'delete_dialects': [1, 6, 8],  ## must be list
              'delete_gender': ['F'],  ## must be list
              'selection': all_info}  ## must be dictionary

dic_input_class = {"select_frames": [3,3], ## must be False, int or list which follows: [before middle frame, after middle frame]
                   'delta': True,  ## must be boolean
                   'double_delta': True}  ## must be boolean

dic_network = {'network_classes': 0, ##expected: int
               'type':[['relu', 'relu', 'relu', 'relu'], [128, 64, 32, 16]], ## supports cnn (fixed), nn (array, array of layers in str, array of size layers in int)
               #'type': 'cnn',
               'batch_size': 125,  ## must be int
               'epoch': 500,
               'weighted': True}  ## must be int

dic_info = {**dic_mfcc, **dic_network, **dic_design, **dic_input_class, "poging": attempt}