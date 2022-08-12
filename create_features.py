from utils import Nonestring
#################################MODEL CHOICE###############################################################################################################################

############################################################################################################################################################################
#######################################DATASET CHOICE#######################################################################################################################

############################################################################################################################################################################
#################################TEXT CHOICE################################################################################################################################

batch_size=1

    
#############################THE FOLLOWING SECTION IS NOT SUPPOSED TO BE MODIFIED BY THE USER###############################################################################
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################

############################################################################################################################################################################
############################################################################################################################################################################
###############OMINBOARD####################################################################################################################################################        

############################################################################################################################################################################
############################################################################################################################################################################

from dataset_number import add_one, id_to_index, PlotsterDataset

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score, jaccard_score, accuracy_score
############################################################################################################################################################################

def main():
    import ast
    import os
    import statistics
    from PIL import Image
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    import torch.nn as nn
    import torch.optim as optim
    file_id = []
    for filename in os.listdir("./posters"):
        file_id.append(filename[7:-4])
    print(file_id[0:100])
    import tmdbsimple as tmdb
    import pandas as pd
    import sys
    from bit_model import KNOWN_MODELS as BiT_MODELS
    import numpy as np
    import torch
    import torchvision.models as models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
############################################################################################################################################################################
##############################DEFINING THE NEURAL NETWORK###################################################################################################################
############################################################################################################################################################################    
    input_resolution = 224
    
    sys.path.insert(1, '/home/romain/Documents/CLIP/CLIP/')
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import clip
    from simple_tokenizer import SimpleTokenizer
    
    clip_model = "RN50"
    from simple_tokenizer import SimpleTokenizer
    
    model = torch.jit.load("/home/romain/Documents/CLIP/CLIP/model.pt").cuda().eval()
    
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()        
    clip_image_model, clip_transform = clip.load(clip_model, device=device, jit=False)
    tokenizer = SimpleTokenizer()
    sys.path.insert(1, '/mnt/HD1/datasets/Plotster/')
    
    
    
    model_name = 'BiT-M-R50x1'
    bit_image_model = BiT_MODELS[model_name](use_fc=False)
    bit_image_model.load_from(np.load('/home/romain/Pretrained/BiT-M-R50x1.npz'))
    bit_image_model.to(device)
    bit_image_model.eval()
    
    bit_transform = Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    
    
    resnet50 = models.resnet50(pretrained=True)
    modules=list(resnet50.children())[:-1]
    RN_image_model=nn.Sequential(*modules).to(device)
    for p in model.parameters():
        p.requires_grad = False
    
    
    RN_transform = Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
    ])
    
    RN_transform_norm = Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
        
        

    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, '/home/romain/Documents/CLIP/CLIP/')
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import clip
    from simple_tokenizer import SimpleTokenizer
    
    clip_model = "RN50"
    from simple_tokenizer import SimpleTokenizer
    
    model = torch.jit.load("/home/romain/Documents/CLIP/CLIP/model.pt").cuda().eval()
    
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()        
    clip_text_model, transform = clip.load(clip_model, device=device, jit=False)
    tokenizer = SimpleTokenizer()
    
    

    from sentence_transformers import SentenceTransformer
    BL_text_model = SentenceTransformer('/home/romain/Downloads/bert_large_CLS')
    BL_text_model.eval()
    
        
        

    from sentence_transformers import SentenceTransformer
    BS_text_model = SentenceTransformer('/home/romain/Downloads/bert_cls')
    BS_text_model.eval()
    
    
    
    ############################################################################################################################################################################
    ############################################################################################################################################################################
    ############################################################################################################################################################################
    
    #Similarity = nn.CosineSimilarity()
    
    dataset = PlotsterDataset(text_file = "list_clear_id_train", transform1 = clip_transform, transform2 = bit_transform, transform3 = RN_transform)
    
        
    train_dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4, drop_last = True, pin_memory=True)
                                
    test_dataset = PlotsterDataset(text_file = "list_clear_id_test", transform1 = clip_transform, transform2 = bit_transform, transform3 = RN_transform)
    
        
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4, drop_last = True, pin_memory=True)
    valid_dataset = PlotsterDataset(text_file = "list_clear_id_valid", transform1 = clip_transform, transform2 = bit_transform, transform3 = RN_transform)
    
        
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4, drop_last = True, pin_memory=True)
    
                                
    
    ############################################################################################################################################################################
    ############################################################################################################################################################################
    ############################################################################################################################################################################
    
    ###########################################TRAINING#########################################################################################################################
    ############################################################################################################################################################################
    for i_batch, sample in enumerate(train_dataloader):
        images_clip, images_bit, images_RN, genres, titles, plots, idx = sample
        
        genre_input = torch.zeros(len(genres), 19, dtype=torch.long)
        count_film = 0
        for genre_list in genres:
            genre_list = ast.literal_eval(genre_list)
            for genre_dict in genre_list:
                genre_input[count_film] = add_one(genre_input[count_film], id_to_index(genre_dict['id']))
            count_film += 1
    
        genre_input = genre_input.to(device)
        
        
        text_tokens = [tokenizer.encode(desc) for desc in plots]
        text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
        sot_token = tokenizer.encoder['<|startoftext|>']
        eot_token = tokenizer.encoder['<|endoftext|>']
        
        for i, tokens in enumerate(text_tokens):
            tokens = [sot_token] + tokens[:model.context_length-2] + [eot_token]
            text_input[i, :len(tokens)] = torch.tensor(tokens)
    
        plots_clip = text_input.cuda()
        
        title_tokens = [tokenizer.encode(desc) for desc in titles]
    
        title_input = torch.zeros(len(title_tokens), model.context_length, dtype=torch.long)
        sot_token = tokenizer.encoder['<|startoftext|>']
        eot_token = tokenizer.encoder['<|endoftext|>']
    
        for i, tokens in enumerate(title_tokens):
            tokens = [sot_token] + tokens[:model.context_length-2] + [eot_token]
            title_input[i, :len(tokens)] = torch.tensor(tokens)
    
        titles_clip = title_input.cuda()
        
        
        image_input_clip = images_clip.cuda()
        image_input_bit = images_bit.cuda()
        image_input_RN = images_RN.cuda()
        
        bit_im_features = bit_image_model(image_input_bit).to(device)
        
        RN_im_features = RN_image_model(image_input_RN).squeeze().to(device)
        
        
        image_multi = clip_image_model.encode_image(image_input_clip)
        image_multi = image_multi.float()                    
        clip_im_features = image_multi.to(device)
        
    
        
        BS_plot_features = torch.tensor(BS_text_model.encode(plots)).to(device)
        BL_plot_features = torch.tensor(BL_text_model.encode(plots)).to(device)
        
        BS_title_features = torch.tensor(BS_text_model.encode(titles)).to(device)
        BL_title_features = torch.tensor(BL_text_model.encode(titles)).to(device)
        
        
        text_multi, text_feat = clip_text_model.encode_text(plots_clip)
        text_multi = text_multi.float() 
        clip_plot_features = text_multi.to(device)
        
        text_multi, text_feat = clip_text_model.encode_text(titles_clip)
        text_multi = text_multi.float() 
        clip_title_features = text_multi.to(device)     
        
        if not os.path.exists('./features/train_features/RN_posters'):
            os.makedirs('./features/train_features/RN_posters')
            os.makedirs('./features/train_features/BiT_posters')
            os.makedirs('./features/train_features/CLIP_posters')
            os.makedirs('./features/train_features/CLIP-T_titles')
            os.makedirs('./features/train_features/CLIP-T_plots')
            os.makedirs('./features/train_features/Bert-small_plots')
            os.makedirs('./features/train_features/Bert-small_titles')
            os.makedirs('./features/train_features/Bert-large_plots')
            os.makedirs('./features/train_features/Bert-large_titles')
            
        torch.save(RN_im_features.squeeze(dim = 0).detach(),'./features/train_features/RN_posters/%s.pt' % (idx[0]) ) 
        torch.save(bit_im_features.squeeze(dim = 0).detach(),'./features/train_features/BiT_posters/%s.pt' % (idx[0]) ) 
        torch.save(clip_im_features.squeeze(dim = 0).detach(),'./features/train_features/CLIP_posters/%s.pt' % (idx[0]) ) 
        torch.save(clip_title_features.squeeze(dim = 0).detach(),'./features/train_features/CLIP-T_titles/%s.pt' % (idx[0]) ) 
        torch.save(clip_plot_features.squeeze(dim = 0).detach(),'./features/train_features/CLIP-T_plots/%s.pt' % (idx[0]) ) 
        torch.save(BL_plot_features.squeeze(dim = 0).detach(),'./features/train_featuress/Bert-large_plots/%s.pt' % (idx[0]) ) 
        torch.save(BL_title_features.squeeze(dim = 0).detach(),'./features/train_features/Bert-large_titles/%s.pt' % (idx[0]) ) 
        torch.save(BS_plot_features.squeeze(dim = 0).detach(),'./features/train_features/Bert-small_plots/%s.pt' % (idx[0]) ) 
        torch.save(BS_title_features.squeeze(dim = 0).detach(),'./featuresP/train_features/Bert-small_titles/%s.pt' % (idx[0]) )
        
        
        
        
        
        
        
            
            
            
    ###########################################VALIDATION#######################################################################################################################
    ############################################################################################################################################################################            
    for i_batch, sample in enumerate(valid_dataloader):
        images_clip, images_bit, images_RN, genres, titles, plots, idx = sample
        
        genre_input = torch.zeros(len(genres), 19, dtype=torch.long)
        count_film = 0
        for genre_list in genres:
            genre_list = ast.literal_eval(genre_list)
            for genre_dict in genre_list:
                genre_input[count_film] = add_one(genre_input[count_film], id_to_index(genre_dict['id']))
            count_film += 1
    
        genre_input = genre_input.to(device)
        
        
        text_tokens = [tokenizer.encode(desc) for desc in plots]
        text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
        sot_token = tokenizer.encoder['<|startoftext|>']
        eot_token = tokenizer.encoder['<|endoftext|>']
        
        for i, tokens in enumerate(text_tokens):
            tokens = [sot_token] + tokens[:model.context_length-2] + [eot_token]
            text_input[i, :len(tokens)] = torch.tensor(tokens)
    
        plots_clip = text_input.cuda()
        
        title_tokens = [tokenizer.encode(desc) for desc in titles]
    
        title_input = torch.zeros(len(title_tokens), model.context_length, dtype=torch.long)
        sot_token = tokenizer.encoder['<|startoftext|>']
        eot_token = tokenizer.encoder['<|endoftext|>']
    
        for i, tokens in enumerate(title_tokens):
            tokens = [sot_token] + tokens[:model.context_length-2] + [eot_token]
            title_input[i, :len(tokens)] = torch.tensor(tokens)
    
        titles_clip = title_input.cuda()
        
        
        image_input_clip = images_clip.cuda()
        image_input_bit = images_bit.cuda()
        image_input_RN = images_RN.cuda()
        
        bit_im_features = bit_image_model(image_input_bit).to(device)
        
        RN_im_features = RN_image_model(image_input_RN).squeeze().to(device)
        
        
        image_multi = clip_image_model.encode_image(image_input_clip)
        image_multi = image_multi.float()                    
        clip_im_features = image_multi.to(device)
        
    
        
        BS_plot_features = torch.tensor(BS_text_model.encode(plots)).to(device)
        BL_plot_features = torch.tensor(BL_text_model.encode(plots)).to(device)
        
        BS_title_features = torch.tensor(BS_text_model.encode(titles)).to(device)
        BL_title_features = torch.tensor(BL_text_model.encode(titles)).to(device)
        
        
        text_multi, text_feat = clip_text_model.encode_text(plots_clip)
        text_multi = text_multi.float() 
        clip_plot_features = text_multi.to(device)
        
        text_multi, text_feat = clip_text_model.encode_text(titles_clip)
        text_multi = text_multi.float() 
        clip_title_features = text_multi.to(device)     
        
        if not os.path.exists('./features/valid_features/RN_posters'):
            os.makedirs('./features/valid_features/RN_posters')
            os.makedirs('./features/valid_features/BiT_posters')
            os.makedirs('./features/valid_features/CLIP_posters')
            os.makedirs('./features/valid_features/CLIP-T_titles')
            os.makedirs('./features/valid_features/CLIP-T_plots')
            os.makedirs('./features/valid_features/Bert-small_plots')
            os.makedirs('./features/valid_features/Bert-small_titles')
            os.makedirs('./features/valid_features/Bert-large_plots')
            os.makedirs('./features/valid_features/Bert-large_titles')
       
        torch.save(clip_im_features.squeeze(dim = 0).detach(),'./features/valid_features/clip_posters/%s.pt' % (idx[0]) ) 
        torch.save(bit_im_features.squeeze(dim = 0).detach(),'./features/valid_features/bit_posters/%s.pt' % (idx[0]) ) 
        torch.save(RN_im_features.squeeze(dim = 0).detach(),'./features/valid_features/RN_posters/%s.pt' % (idx[0]) ) 
        torch.save(clip_title_features.squeeze(dim = 0).detach(),'./features/valid_features/clip_titles/%s.pt' % (idx[0]) ) 
        torch.save(clip_plot_features.squeeze(dim = 0).detach(),'./features/valid_features/clip_plots/%s.pt' % (idx[0]) ) 
        torch.save(BL_plot_features.squeeze(dim = 0).detach(),'./features/valid_features/bert_large_plots/%s.pt' % (idx[0]) ) 
        torch.save(BL_title_features.squeeze(dim = 0).detach(),'./features/valid_features/bert_large_titles/%s.pt' % (idx[0]) ) 
        torch.save(BS_plot_features.squeeze(dim = 0).detach(),'./features/valid_features/bert_small_plots/%s.pt' % (idx[0]) ) 
        torch.save(BS_title_features.squeeze(dim = 0).detach(),'./features/valid_features/bert_small_titles/%s.pt' % (idx[0]) )     
          
                
                
                
                
                    
    ###########################################TEST##########################################################################################################################
    ############################################################################################################################################################################
    for i_batch, sample in enumerate(test_dataloader):
        images_clip, images_bit, images_RN, genres, titles, plots, idx = sample
        
        genre_input = torch.zeros(len(genres), 19, dtype=torch.long)
        count_film = 0
        for genre_list in genres:
            genre_list = ast.literal_eval(genre_list)
            for genre_dict in genre_list:
                genre_input[count_film] = add_one(genre_input[count_film], id_to_index(genre_dict['id']))
            count_film += 1
    
        genre_input = genre_input.to(device)
        
        
        text_tokens = [tokenizer.encode(desc) for desc in plots]
        text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
        sot_token = tokenizer.encoder['<|startoftext|>']
        eot_token = tokenizer.encoder['<|endoftext|>']
        
        for i, tokens in enumerate(text_tokens):
            tokens = [sot_token] + tokens[:model.context_length-2] + [eot_token]
            text_input[i, :len(tokens)] = torch.tensor(tokens)
    
        plots_clip = text_input.cuda()
        
        title_tokens = [tokenizer.encode(desc) for desc in titles]
    
        title_input = torch.zeros(len(title_tokens), model.context_length, dtype=torch.long)
        sot_token = tokenizer.encoder['<|startoftext|>']
        eot_token = tokenizer.encoder['<|endoftext|>']
    
        for i, tokens in enumerate(title_tokens):
            tokens = [sot_token] + tokens[:model.context_length-2] + [eot_token]
            title_input[i, :len(tokens)] = torch.tensor(tokens)
    
        titles_clip = title_input.cuda()
        
        
        image_input_clip = images_clip.cuda()
        image_input_bit = images_bit.cuda()
        image_input_RN = images_RN.cuda()
        
        bit_im_features = bit_image_model(image_input_bit).to(device)
        
        RN_im_features = RN_image_model(image_input_RN).squeeze().to(device)
        
        
        image_multi = clip_image_model.encode_image(image_input_clip)
        image_multi = image_multi.float()                    
        clip_im_features = image_multi.to(device)
        
    
        
        BS_plot_features = torch.tensor(BS_text_model.encode(plots)).to(device)
        BL_plot_features = torch.tensor(BL_text_model.encode(plots)).to(device)
        
        BS_title_features = torch.tensor(BS_text_model.encode(titles)).to(device)
        BL_title_features = torch.tensor(BL_text_model.encode(titles)).to(device)
        
        
        text_multi, text_feat = clip_text_model.encode_text(plots_clip)
        text_multi = text_multi.float() 
        clip_plot_features = text_multi.to(device)
        
        text_multi, text_feat = clip_text_model.encode_text(titles_clip)
        text_multi = text_multi.float() 
        clip_title_features = text_multi.to(device)     
      
        
       
        if not os.path.exists('./features/valid_features/RN_posters'):
            os.makedirs('./features/test_features/RN_posters')
            os.makedirs('./features/test_features/BiT_posters')
            os.makedirs('./features/test_features/CLIP_posters')
            os.makedirs('./features/test_features/CLIP-T_titles')
            os.makedirs('./features/test_features/CLIP-T_plots')
            os.makedirs('./features/test_features/Bert-small_plots')
            os.makedirs('./features/test_features/Bert-small_titles')
            os.makedirs('./features/test_featuress/Bert-large_plots')
            os.makedirs('./features/test_features/Bert-large_titles')
        
        torch.save(clip_im_features.squeeze(dim = 0).detach(),'./features/test_features/clip_posters/%s.pt' % (idx[0]) ) 
        torch.save(bit_im_features.squeeze(dim = 0).detach(),'./features/test_features/bit_posters/%s.pt' % (idx[0]) )     
        torch.save(RN_im_features.squeeze(dim = 0).detach(),'./features/test_features/RN_posters/%s.pt' % (idx[0]) )  
        torch.save(clip_title_features.squeeze(dim = 0).detach(),'./features/test_features/clip_titles/%s.pt' % (idx[0]) ) 
        torch.save(clip_plot_features.squeeze(dim = 0).detach(),'./features/test_features/clip_plots/%s.pt' % (idx[0]) ) 
        torch.save(BL_plot_features.squeeze(dim = 0).detach(),'./features/test_features/bert_large_plots/%s.pt' % (idx[0]) ) 
        torch.save(BL_title_features.squeeze(dim = 0).detach(),'./features/test_features/bert_large_titles/%s.pt' % (idx[0]) ) 
        torch.save(BS_plot_features.squeeze(dim = 0).detach(),'./features/test_features/bert_small_plots/%s.pt' % (idx[0]) ) 
        torch.save(BS_title_features.squeeze(dim = 0).detach(),'./features/test_features/bert_small_titles/%s.pt' % (idx[0]) ) 
                
                    
                    
               
                       
           
                        
                        
                        
                        
                       
