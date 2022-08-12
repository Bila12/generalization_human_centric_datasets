from utils import Nonestring
import statistics
#################################MODEL CHOICE###############################################################################################################################
###Choose image models from BiT, CLIP, RN50 
###Choose language models from Bert-small, Bert-large, CLIP-T
for language_model in ["Bert-large"]:
    for vision_model in ["CLIP","BiT", "RN50"]:
        if language_model is not None or vision_model is not None:
            text_to_encode = "title"
            filtered = False
            batch_size=50
            text_to_encode_mem = text_to_encode
            max_epoch = 300
            weighted_accuracies = []
            f1_scores = []
            for i in range(5):
            
                
                #############################THE FOLLOWING SECTION IS NOT SUPPOSED TO BE MODIFIED BY THE USER###############################################################################
                ############################################################################################################################################################################
                ############################################################################################################################################################################
                ############################################################################################################################################################################
                if filtered == True:
                        filteredString = "filtered"
                else:
                        filteredString = None
                
                if vision_model is not None:
                    isImUsed = "im"
                else:
                    isImUsed = None
                if language_model is None:
                    text_to_encode = None
                ############################################################################################################################################################################
                ############################################################################################################################################################################
                ###############OMINBOARD####################################################################################################################################################        
                
                ############################################################################################################################################################################
                ############################################################################################################################################################################
                from feature_dataset import add_one, id_to_index, PlotsterDataset
                
                from torch.utils.data import DataLoader
                from sklearn.metrics import f1_score, average_precision_score, accuracy_score
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
                    #print(file_id[0:100])
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
                    if vision_model == "CLIP":
                        im_dim = 1024
                       
                    
                    
                    elif vision_model == "BiT":
                        im_dim = 2048
                    
                    elif vision_model == "RN50":
                        im_dim = 2048
                        
                    else:
                        image_model = None
                        transform = None
                        im_dim = 0
                        
                    if language_model == "CLIP-T":
                        la_dim = 1024
                    
                    elif language_model == "Bert-large":
                        la_dim = 1024
                        
                        
                    elif language_model == "Bert-small":
                        la_dim = 768
                    else:
                        text_model = None
                        la_dim = 0
                    
                    dense_to_genre_1 = nn.Linear(im_dim+la_dim, 256).to(device)
                    dense_to_genre_2 = nn.Linear(256, 19).to(device)
                    #dense_to_genre_1.load_state_dict(torch.load('/mnt/SSD/datasets/Plotster/output/models/dense_to_genre_1_BiT_Bert_title_BCE_plot.ckpt'))
                    #dense_to_genre_2.load_state_dict(torch.load('/mnt/SSD/datasets/Plotster/output/models/dense_to_genre_2_BiT_Bert_title_BCE_plot.ckpt'))
                    
                    ############################################################################################################################################################################
                    ############################################################################################################################################################################
                    ############################################################################################################################################################################
                    
                    activation = nn.ReLU().to(device)
                    sigmo = nn.Sigmoid().to(device)
                    criterion = nn.MSELoss().to(device)
                    params = list(dense_to_genre_1.parameters()) + list(dense_to_genre_2.parameters())
                    optimizer = optim.Adam(params, lr=0.0001)
                    
                    #Similarity = nn.CosineSimilarity()
                    
                    
                    dataset = PlotsterDataset(ttv = "train", filtered = filtered, im_model = vision_model, txt_model = language_model, txt = text_to_encode_mem + "s")
                    
                    test_dataset = PlotsterDataset(ttv = "test", filtered = filtered, im_model = vision_model, txt_model = language_model, txt = text_to_encode_mem + "s")
                    
                    valid_dataset = PlotsterDataset(ttv = "valid", filtered = filtered, im_model = vision_model, txt_model = language_model, txt = text_to_encode_mem + "s")
                        
                    
                    
                    
                     
                    dataloader = DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=10, drop_last = True, pin_memory = True)
                                                
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=10, drop_last = True, pin_memory = True)
                                                
                                                
                    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=10, drop_last = True, pin_memory = True)
                    #print(len(dataloader))
                    #print(len(test_dataloader))
                    #print(len(valid_dataloader))
                    #torch.multiprocessing.set_start_method('spawn',force=True)
                    ############################################################################################################################################################################
                    ############################################################################################################################################################################
                    ############################################################################################################################################################################
                    epoch = 0
                    losses = []
                    while epoch <max_epoch:
                        
                        step = 0  
                           
                    ###########################################TRAINING#########################################################################################################################
                    ############################################################################################################################################################################
                        for i_batch, sample in enumerate(dataloader):
                            im_features, text_features, genres = sample
                            
                            genre_input = torch.zeros(len(genres), 19, dtype=torch.long)
                            count_film = 0
                            for genre_list in genres:
                                genre_list = ast.literal_eval(genre_list)
                                for genre_dict in genre_list:
                                    genre_input[count_film] = add_one(genre_input[count_film], id_to_index(genre_dict['id']))
                                count_film += 1
                        
                            genre_input = genre_input.to(device)
                            
                            
                            
                            if vision_model is None:
                                features = text_features.to(device)
                            elif language_model is None:
                                features = im_features.to(device)
                            else :    
                                features = torch.cat((text_features, im_features), dim = 1).to(device)
                            
                            
                            
                            
                            genre_predicted = sigmo(dense_to_genre_2(activation(dense_to_genre_1(features))))
                            optimizer.zero_grad()
                            
                            genre_predicted_log = torch.log(genre_predicted).to(device)
                            genre_predicted_log_one = torch.log(1-genre_predicted).to(device)
                            
                            
                            loss = -(torch.mul(genre_input, genre_predicted_log).sum() + torch.mul(1-genre_input, genre_predicted_log_one).sum())
                            
                            genre_predicted = genre_predicted.round().type(torch.LongTensor).to(device)
                            
                            
                            w_acc = ((torch.mul(1-genre_input, 1-genre_predicted).sum().item()/batch_size)/((1-genre_input).sum().item()/batch_size)+(torch.mul(genre_input, genre_predicted).sum().item()/batch_size)/(genre_input.sum().item()/batch_size))/2
                            
                            
                            
                            loss.backward()
                            optimizer.step()
                            '''
                            if step % 100 ==  0:
                                print("epoch = " + str(epoch))
                                print("step = " + str(step))
                                print("mean_proportion_of_zero: ", end ="")
                                print((19*batch_size-genre_input.sum().item())/batch_size/19)
                                print("mean_true_prediction: ", end ="")
                                print(genre_predicted.eq(genre_input).sum().item()/batch_size/19)
                                print("num_true_positive_per_film: ", end ="")
                                print(torch.mul(genre_input, genre_predicted).sum().item()/batch_size)
                                print("num_positive_per_film: ", end ="")
                                print(genre_input.sum().item()/batch_size)
                                print("true_positive_over_positive_per_film: ", end ="")
                                print((torch.mul(genre_input, genre_predicted).sum().item()/batch_size)/(genre_input.sum().item()/batch_size))
                                print("weighted accuracy: ", end ="")
                                print(((torch.mul(1-genre_input, 1-genre_predicted).sum().item()/batch_size)/((1-genre_input).sum().item()/batch_size)+(torch.mul(genre_input, genre_predicted).sum().item()/batch_size)/(genre_input.sum().item()/batch_size))/2)
                            '''
                            
                            
                            
                            
                    ###########################################VALIDATION#######################################################################################################################
                    ############################################################################################################################################################################            
                            if len(dataloader)-1==step:
                                loss_valid = []
                                val_step = 0
                                for i_batch, sample in enumerate(valid_dataloader):
                                    im_features, text_features, genres = sample
                            
                                    genre_input = torch.zeros(len(genres), 19, dtype=torch.long)
                                    count_film = 0
                                    for genre_list in genres:
                                        genre_list = ast.literal_eval(genre_list)
                                        for genre_dict in genre_list:
                                            genre_input[count_film] = add_one(genre_input[count_film], id_to_index(genre_dict['id']))
                                        count_film += 1
                                
                                    genre_input = genre_input.to(device)
                                    
                                    
                                    
                                    if vision_model is None:
                                        features = text_features.to(device)
                                    elif language_model is None:
                                        features = im_features.to(device)
                                    else :    
                                        features = torch.cat((text_features, im_features), dim = 1).to(device)
                                    
                                    
                                    
                                    
                                    genre_predicted = sigmo(dense_to_genre_2(activation(dense_to_genre_1(features))))
                                    optimizer.zero_grad()
                                    
                                    genre_predicted_log = torch.log(genre_predicted).to(device)
                                    genre_predicted_log_one = torch.log(1-genre_predicted).to(device)
                                    
                                    
                                    loss = -(torch.mul(genre_input, genre_predicted_log).sum() + torch.mul(1-genre_input, genre_predicted_log_one).sum())
                                    
                                    genre_predicted = genre_predicted.round().type(torch.LongTensor).to(device)
                                    
                                    
                                    
                                    w_acc = ((torch.mul(1-genre_input, 1-genre_predicted).sum().item()/batch_size)/((1-genre_input).sum().item()/batch_size)+(torch.mul(genre_input, genre_predicted).sum().item()/batch_size)/(genre_input.sum().item()/batch_size))/2
                                    
                                    
                                    
                                    
                                    
                                    loss_valid.append(loss.cpu().detach().numpy().item())
                                    
                                    val_step += 1
                                    
                                losses.append(statistics.mean(loss_valid))
                                
                                if len(losses)<=10:
                                    torch.save(dense_to_genre_1.state_dict(),'%s/dense_to_genre_1_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), epoch, i))                
                                    torch.save(dense_to_genre_2.state_dict(),'%s/dense_to_genre_2_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), epoch, i))   
                                    
                                elif losses.index(min(losses)) != len(losses)-10:
                                    torch.save(dense_to_genre_1.state_dict(),'%s/dense_to_genre_1_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), epoch, i))                
                                    torch.save(dense_to_genre_2.state_dict(),'%s/dense_to_genre_2_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), epoch, i))
                                    os.remove('%s/dense_to_genre_1_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), epoch-10, i))
                                    os.remove('%s/dense_to_genre_2_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), epoch-10, i))  
                                    epoch_mem = epoch 
                                    
                                else:                    
                                    dense_to_genre_1.load_state_dict(torch.load('%s/dense_to_genre_1_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), epoch-9, i)))
                                    dense_to_genre_2.load_state_dict(torch.load('%s/dense_to_genre_2_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), epoch-9, i)))
                                    epoch_mem = epoch-9
                                    for e in [epoch - k for k in [10, 8, 7, 6, 5, 4, 3, 2, 1]]:
                                        os.remove('%s/dense_to_genre_1_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), e, i))
                                        os.remove('%s/dense_to_genre_2_feat_%s%s%s%sBCE_plotster_epoch_%d_loop_%d.ckpt' % ('./output/models',Nonestring(vision_model), Nonestring(language_model), Nonestring(text_to_encode), Nonestring(filteredString), e, i))    
                                    
                                    epoch = max_epoch-1
                                    
                                    
                    ###########################################TESTING##########################################################################################################################
                    ############################################################################################################################################################################
                                    
                                    
                                    
                                if epoch == max_epoch-1 :
                                    loss_test_f1 = []
                                    loss_test_auc = []
                                    loss_test = []
                                    val_step = 0
                                       
                                    for i_batch, sample in enumerate(test_dataloader):
                                        im_features, text_features, genres = sample
                            
                                        genre_input = torch.zeros(len(genres), 19, dtype=torch.long)
                                        count_film = 0
                                        for genre_list in genres:
                                            genre_list = ast.literal_eval(genre_list)
                                            for genre_dict in genre_list:
                                                genre_input[count_film] = add_one(genre_input[count_film], id_to_index(genre_dict['id']))
                                            count_film += 1
                                    
                                        genre_input = genre_input.to(device)
                                        
                                        
                                        
                                        if vision_model is None:
                                            features = text_features.to(device)
                                        elif language_model is None:
                                            features = im_features.to(device)   
                                        else :    
                                            features = torch.cat((text_features, im_features), dim = 1).to(device)
                                        
                                        
                                        
                                        
                                        genre_predicted = sigmo(dense_to_genre_2(activation(dense_to_genre_1(features))))
                                        optimizer.zero_grad()
                                        
                                        genre_predicted_log = torch.log(genre_predicted).to(device)
                                        genre_predicted_log_one = torch.log(1-genre_predicted).to(device)
                                        
                                        
                                        loss = -(torch.mul(genre_input, genre_predicted_log).sum() + torch.mul(1-genre_input, genre_predicted_log_one).sum())
                                        
                                        genre_predicted_rounded = genre_predicted.round().type(torch.LongTensor).to(device)
                                        
                                        
                                        
                                        w_acc = ((torch.mul(1-genre_input, 1-genre_predicted_rounded).sum().item()/batch_size)/((1-genre_input).sum().item()/batch_size)+(torch.mul(genre_input, genre_predicted_rounded).sum().item()/batch_size)/(genre_input.sum().item()/batch_size))/2
                                        
                                       
                                        
                                        
                                        loss_test_f1.append(f1_score(genre_input.cpu(),genre_predicted_rounded.cpu(), average="samples"))
                                        loss_test_auc.append(average_precision_score(genre_input.cpu(),genre_predicted_rounded.cpu(), average="samples"))
                                        loss_test.append(((torch.mul(1-genre_input, 1-genre_predicted_rounded).sum().item()/batch_size)/((1-genre_input).sum().item()/batch_size)+(torch.mul(genre_input, genre_predicted_rounded).sum().item()/batch_size)/(genre_input.sum().item()/batch_size))/2)
                                        
                                        val_step += 1
                                    
                                    f1_scores.append(statistics.mean(loss_test_f1))
                                    weighted_accuracies.append(statistics.mean(loss_test))
                                           
                            step += 1
                            
                            
                        epoch += 1
            print("f1 scores for", "%s%s%s%s%sclassification_features:" % (Nonestring(vision_model), Nonestring(language_model), Nonestring(isImUsed), Nonestring(text_to_encode), Nonestring(filteredString) ) , f1_scores) 
            print("weighted accuracies for", "%s%s%s%s%sclassification_features:" % (Nonestring(vision_model), Nonestring(language_model), Nonestring(isImUsed), Nonestring(text_to_encode), Nonestring(filteredString) ) , weighted_accuracies)           
            print("average f1 score:", statistics.mean(f1_scores))
            print("standard deviation in f1 score:", statistics.stdev(f1_scores))
            print("average weighted accuracy:", statistics.mean(weighted_accuracies))
            print("standard deviation in weighted accuracy score:", statistics.stdev(weighted_accuracies))
            
