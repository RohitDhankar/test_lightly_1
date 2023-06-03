# conda activate tensorflow_gpuenv

#GIT... Practical Deep Learn -- https://github.com/PracticalDL/Practical-Deep-Learning-Book
#BOOK...Practical Deep Learning for Cloud, Mobile, and Edge - by Anirudh Koul, Siddha Ganju and Meher Kasam. 
#SOURCE..https://github.com/PracticalDL/Practical-Deep-Learning-Book/blob/master/code/chapter-4/2-similarity-search-level-1.ipynb

## practical rm -rf caltech101/BACKGROUND_Google model = InceptionV3(weights='imagenet',
## find DIR_PATH/ -print0 | sed -nz '1~3p' | xargs -0 cp --target-dir=DIR_PATH/

#Image Transforms 
import pandas as pd
from img_transforms import get_cntrs , draw_cntrs , save_img_cntrs #get_hsv_img
# Nearest Neighbours 
import tensorflow , numpy as np , os , pickle , time
from tqdm import tqdm, tqdm_notebook
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import PIL , matplotlib , glob
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import norm

from tensorflow.keras.preprocessing import image as tf_image_prep
##https://keras.io/api/layers/preprocessing_layers/image_preprocessing/resizing/#resizing-class

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tf.keras.preprocessing import image
# from tf.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D

import argparse
parser = argparse.ArgumentParser(description='py_knn_img_similarity.')
parser.add_argument('--root_dir', help='root_dir_imgs', nargs='?', const=0)
parser.add_argument('--perc_split', help='train_test_split', nargs='?', const=0)
parser.add_argument('--data_type_flag', help='dataType', nargs='?', const=0)

# # args = parser.parse_args()
# # root_dir_1 = str(args.root_dir_1)
# # root_dir_1 = "/"
# # data_type_flag = str(args.data_type_flag)

# ## Earlier >> #model = ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))

# """
# CNN -- KERAS 
# 1st -->> feature extractor,
# 2nd -->> image classifier,
# Thus -- include_top=False, will DROP the Classifier 
# TRANSFER LEARNING -- https://research.cs.wisc.edu/machine-learning/shavlik-group/torrey.handbook09.pdf
# """

def model_picker(name):
    if (name == 'vgg16'):
        model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3), # ints (img_height, img_width ,channel count)
                      pooling='max')
    elif (name == 'vgg19'):
        model = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3), # ints (img_height, img_width ,channel count)
                      pooling='max')
    elif (name == 'mobilenet'):
        # TODO -- MobileNet -- extra PARAMS -- depth_multiplier , alpha
        model = MobileNet(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3), # ints (img_height, img_width ,channel count)
                          pooling='max',
                          depth_multiplier=1,
                          alpha=1)
    elif (name == 'inception'):
        model = InceptionV3(weights='imagenet',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='max')
    elif (name == 'resnet'):
        model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3), # ints (img_height, img_width ,channel count)
                        pooling='max')
    elif (name == 'xception'):
        model = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3), # ints (img_height, img_width ,channel count)
                         pooling='max')
    else:
        print("Specified model not available")
    print("----Type(model---",type(model)) #<class 'keras.engine.functional.Functional'>
    return model


def extract_features(img_path, model):
    """
    https://keras.io/api/layers/preprocessing_layers/image_preprocessing/resizing/#resizing-class
    
    """
    input_shape = (224, 224, 3) # ints (img_height, img_width ,channel count)
    img = tf_image_prep.load_img(img_path, target_size=(input_shape[0], input_shape[1])) # tf_image
    img_array = tf_image_prep.img_to_array(img)
    print("-[INFO_extract_features]---type(img_array-->>",type(img_array)) #<class 'numpy.ndarray'>
    #
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    print("-[INFO_extract_features]---type(preprocessed_img-->>",type(preprocessed_img)) #<class 'numpy.ndarray'>
    #
    features = model.predict(preprocessed_img)
    print("-[INFO_extract_features]---type(features-->>",type(features)) #<class 'numpy.ndarray'>
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    # print("type(normalized_features)---",type(normalized_features))
    print("---normalized_features.shape---",normalized_features.shape)
    # print("---normalized_features.ndim---",normalized_features.ndim)
    # print("len(normalized_features)-----",len(normalized_features))
    #print("normalized_features)-----",normalized_features[0:5])
    return normalized_features



def get_file_list(root_dir):
    """
    
    """
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
    print("-[INFO_get_file_list]-file_list--",file_list[0:5])
    return sorted(file_list)


# def classname(str):
#     """
#     # Helper function to get the classname
#     """
#     return str.split('/')[-2]


# def classname_filename(str):
#     """
#     # Helper function to get the classname and filename
#     """
#     return str.split('/')[-2] + '/' + str.split('/')[-1]


# def plot_nn_images(ls_similar_image_paths ,ls_distances,root_dir_label_name):
#     """
#     # Helper functions to plot the nearest images given a query image
#     """
#     import shutil , os 
#     # from datetime import datetime
#     # dt_time_now = datetime.now()
#     # hour_now = dt_time_now.strftime("_%m_%d_%Y_%H")

#     ls_similar_img_name = []
#     ls_copy_to_path = []
#     ls_k_near_img = []

#     meta_data_df = pd.DataFrame()
#     for iter_k in range(len(ls_similar_image_paths)):
#         for similar_img_name in ls_similar_image_paths[iter_k]:
#             print("--plot_nn_images-similar_image_paths--NOW-->>ls_similar_image_paths[iter_k]--",ls_similar_image_paths[iter_k])
#             print("---plot_nn_images--similar_img_name---",similar_img_name)
#             k_near_img = similar_img_name.split('/')[-1] 
#             print("---plot_nn_images--k_near_img--->>\n",k_near_img)
#             init_query_img = ls_similar_image_paths[iter_k][0].split('/')[-1] 
#             print("---plot_nn_images--init_query_img--->>\n",init_query_img) # TODO -- main -- Copy to a DIR NAME with Minutes and -MAIN QUERY IMAGE FileName
#             query_img = init_query_img.split('_')[0] 
#             print("---plot_nn_images--query_img---->>\n",query_img)

#             dt_time_now = datetime.now() 
#             #hour_now = dt_time_now.strftime("_%m_%d_%Y_%H_/")
#             query_img_dir_path = "./output_dir/knn_similar_images/"+str(root_dir_label_name)+"/"+str(query_img)+"/"
#             print("---query_img_dir_path----\n",query_img_dir_path)
#             #hourly_dir = os.path.join(crop_img_dir_path)
#             if not os.path.exists(query_img_dir_path):
#                 os.makedirs(query_img_dir_path)
#             print("---COPY THIS ---",similar_img_name)
#             copy_to_path = str(query_img_dir_path)+str(k_near_img)
#             print("---COPY HERE ----copy_to_path----",copy_to_path)
#             shutil.copyfile(similar_img_name, copy_to_path)
#             ls_copy_to_path.append(copy_to_path)
#             ls_similar_img_name.append(similar_img_name)
#             ls_k_near_img.append(k_near_img)
#             #
#         meta_data_df["similar_img_name"] = ls_similar_img_name
#         meta_data_df["copy_to_path"] = ls_copy_to_path
#         meta_data_df["k_near_img"] = ls_k_near_img
        
#         output_path='meta_data_df_1.csv'
#         meta_data_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

#     return meta_data_df # for writing to the Meta Data DF in separate call 



#     #second_now = dt_time_now.strftime("%m_%d_%Y_%H_%M_%S_/") 
#     # #minute_now = dt_time_now.strftime("%m_%d_%Y_%H_%M_/") 
#     #     images.append(mpimg.imread(similar_img_name))
#     # plt.figure(figsize=(20, 10))
#     # columns = 4
#     #print("---plot_nn_images--len(images----",len(images))
#     #print("---plot_nn_images--type(images[0]----",type(images[0])) ## First Image -- ndArray

#     # # for iter_n, image in enumerate(images):
#     # #     print("---plot_nn_images-->>iter_n---",iter_n)
#     # #     print("---distances ----",distances[iter_n])

#     # #     ax = plt.subplot(len(images) / columns + 1, columns, iter_n + 1)
#     # #     if iter_n == 0:
#     # #         print("----MAIN QUERY IMAGE FileName --->> ",similar_image_paths[iter_n])
#     # #         ax.set_title("---Query Image----")# + classname_filename(similar_image_paths[iter_n]))
#     # #         plt.imshow(image)
#     # #         plt.gcf().set_dpi(300)
#     # #         plt.show()#block=False)


#     # #         ### TODO -- All below here 
#     # #         #ax.set_title("Query Image\n" + classname_filename(similar_image_paths[iter_n]))
#     # #         # init_QueryImgName = classname_filename(similar_image_paths[iter_n]) # TODO -'out_put_dir/_Faces/image_0015.jpg_.pdf'
#     # #         # img_className = init_QueryImgName.split('/')[0]
#     # #         # img_fileName = init_QueryImgName.split('/')[1]
#     #         # print("--plot_nn_images-img_className-",img_className)
#     #         # print("--plot_nn_images-img_fileName-",img_fileName)

#     #         # img_className_dir_path = '../output_dir/_QueryImg_ClassName_/'+str(img_className)
#     #         # #hourly_save_dir = os.path.join(f'{hour_now}',img_className_dir_path)
#     #         # if not os.path.exists(img_className_dir_path):
#     #         #     os.makedirs(img_className_dir_path)
#     #         # plt.imsave(str(img_className_dir_path)+"/"+str(img_fileName)+"_.png", image)  #TODO -- Fix this getting 2KB PDF Files
#     #     else:
#     #         ax.set_title("Similar Image")
#     #         plt.imshow(image)
#     #         plt.gcf().set_dpi(300)
#     #         plt.show()#block=False)


#     #         #ax.set_title("Similar Image\n" + classname_filename(similar_image_paths[iter_n]) + "\nDistance: " + str(float("{0:.2f}".format(distances[iter_n]))))
#     #         ### TODO -- All below here 
#     #         # init_QueryImgName = classname_filename(similar_image_paths[iter_n]) # TODO -'out_put_dir/_Faces/image_0015.jpg_.pdf'
#     #         # img_className = init_QueryImgName.split('/')[0]
#     #         # img_fileName = init_QueryImgName.split('/')[1]
            
#     #         # img_className_dir_path = '../output_dir/_SimilarImg_ClassName_/'+str(img_className)+'_nn_distance_'+ str(float("{0:.2f}".format(distances[iter_n])))
#     #         # if not os.path.exists(img_className_dir_path):
#     #         #     os.makedirs(img_className_dir_path)
#     #         # plt.imsave(str(img_className_dir_path)+str(img_fileName)+"_.png", image) 

        
#     #     ### TODO -- All below here 
#     #     # plt.imshow(image)
#     #     # plt.gcf().set_dpi(300)
#     #     # plt.show()#block=False)

#     #     #plt.savefig(str(hourly_plots_dir)+"/"+str(pdf_fileName)+"_.pdf", format='pdf', dpi=300)
#     #     # To save the plot in a high definition format i.e. PDF, uncomment the following line:
#     #     #plt.savefig('results/' + str(random.randint(0,10000))+'.pdf', format='pdf', dpi=1000)
#     #     # We will use this line repeatedly in our code.


def get_all_files_df(root_dir,root_dir_label_name,file_extn_type=None):
    file_extn_type = ".png"
    file_names_list = []
    file_sub_dirs_list = []

    for (dir_paths, dirs, files) in os.walk(root_dir):
        for file_to_be_found in files:
            if file_to_be_found.endswith(file_extn_type):
                #file_list.append(os.path.join(paths, file))
                file_names_list.append(file_to_be_found)
                file_sub_dirs_list.append(os.path.join(dir_paths,file_to_be_found))
    df_img_paths = pd.DataFrame({'img_file_name':file_names_list,'sub_dir_path':file_sub_dirs_list,})
    print("----shape->>-df_img_paths--",df_img_paths.shape)
    df_img_paths["file_extn_type"]= file_extn_type
    df_img_paths.to_csv('df_img_paths.csv',mode='a',index=False) #
    #TODO-- mode Append careful large bloated DF >> CSV 



# def query_img_neighbors(ls_ftrs,ls_fileNames,neighbors,root_dir_label_name):
#     """
#     TODO - experiment diff distances -Minkowski,Manhattan,Jaccardian,weighted Euclidean 
#     (weight is contribution of each feature -->> pca.explained_variance_ratio_ )
#     """
#     print("---query_img_neighbors----aa----LEN(ls_ftrs----",len(ls_ftrs))
#     print("---query_img_neighbors----aa----LEN(ls_fileNames----",len(ls_fileNames))
#     ls_similar_image_paths = []
#     ls_distances = []

#     num_images = 500 #TODO --Not defined in BOOK Code - so ts TOTAL IMAGES Count ??
#     for iter_k in range(22):
#         print("-query_img_neighbors--iter---",iter_k)
#         random_image_index = random.randint(0, num_images) # picks any Random Img Indx -- max val == num_images -- this changes on every run . 
#         print("---random_image_index--",random_image_index)
#         #random_image_index = 50 
        
#         distances, indices = neighbors.kneighbors([ls_ftrs[random_image_index]])
#         print("--query_img_neighbors---distances--",distances)
#         print("--query_img_neighbors---indices--",indices)
#         # # Don't take the first closest image as it will be the same image
#         # print("----Query_Image--Same IMAGE--",ls_fileNames[indices[0][iter_k]])
#         similar_image_paths = [ls_fileNames[random_image_index]] + [ls_fileNames[indices[0][iter_j]] for iter_j in range(1, 4)] # 4
#         print("---len(similar_image_paths--",len(similar_image_paths))
#         print("---similar_image_paths--",similar_image_paths)
#         ls_similar_image_paths.append(similar_image_paths)
#         print("----distances[0]----",distances[0])
#         ls_distances.append(distances[0])
#     return ls_similar_image_paths ,ls_distances


def get_fileNames_pickle(root_dir_1,str_cnst_caltech):
    """
    probably no need to Pickle ? 
    
    """
    #write Pickle Files 
    print("--get_fileNames_pickle----str_cnst_caltech-",str_cnst_caltech)
    print("--get_fileNames_pickle----root_dir_1-",root_dir_1)
    path_pickle = "/home/dhankar/temp/11_22/a______lightly_2/datasets/output_dir/pickle_files/"
    
    ls_fileNames = get_file_list(root_dir_1)
    #sorted(get_file_list(root_dir_1)) ## Not Reqd
    #pickle.dump(ls_fileNames, open('./datasets/output_dir/pickle_files/'+str(str_cnst_caltech)+'_ls_fileNames_.pickle','wb'))
    #pickle.dump(ls_fileNames, open('./datasets/output_dir/pickle_files/file_1_ls_fileNames_.pickle','wb'))
    #pickle.dump(ls_fileNames, open(path_pickle+'file_1_ls_fileNames_.pickle','wb'))
    return ls_fileNames

def get_img2vec(ls_fileNames,str_cnst_caltech,model_archType,root_dir_label_name):
    """
    
    """
    print("-[INFO_get_img2vec]-STARTED--Extracting Features-->>")
    print("---get_img2vec---len(ls_fileNames)-",len(ls_fileNames))

    ls_feature = []
    #for iter_k in tqdm(range(len(ls_fileNames))): 
    for iter_k in tqdm(range(5)): #TODO -- Check and manually delete -->> ls_fileNames--ABOVE 
        #TODO -- hardcoded as 500 for test--#for i in tqdm(range(500)):
        print("--[INFO_get_img2vec]-STARTED---Extracting Contours ----->>",iter_k)


        contours, hierarchy , img_init , max_cntr_area , max_cntr_perimeter = get_cntrs(ls_fileNames[iter_k])#,img_path)
        if isinstance(contours, str):
            print("--No cntrs--ERROR in get_cntrs---->>--type(contours)-",type(contours))
            pass
        else:
            file_name_img_cntrs = str(ls_fileNames[iter_k]).split('/')[-1]
            #print("---type--cntrs----",type(contours)) # Lists TODO - dont draw - get validation from Contours 
            print("---LEN-cntrs----",len(contours))
            
            # Only Draw -- max_cntr_area 
            img_max_area_cntr , img_cntrs_type = draw_cntrs(max_cntr_area,img_init) # 
            print("--TEST---aa---type(img_max_area_cntr)----",type(img_max_area_cntr))
            img_cntrs_type = "max_area_cntr"
            path_img_cntrs = save_img_cntrs(img_max_area_cntr,root_dir_label_name,file_name_img_cntrs,img_cntrs_type)
            
            # Only Draw -- max_cntr_perimeter
            max_cntr_perimeter = max_cntr_perimeter[0] ## Its a SORTED List of Numpy Arrays 
            img_max_perimeter_cntr , img_cntrs_type = draw_cntrs(max_cntr_perimeter,img_init) # 
            print("--TEST---aa---type(img_max_perimeter_cntr)----",type(img_max_perimeter_cntr))
            img_cntrs_type = "max_perimeter_cntr"
            path_img_cntrs = save_img_cntrs(img_max_perimeter_cntr,root_dir_label_name,file_name_img_cntrs,img_cntrs_type)
            
            # Draw All Countours 
            img_cntrs , img_cntrs_type = draw_cntrs(contours,img_init) # If Lists LEN ==0 TODO - dont draw - get validation from Contours 
            img_cntrs_type = "all_cntrs"
            path_img_cntrs = save_img_cntrs(img_cntrs,root_dir_label_name,file_name_img_cntrs,img_cntrs_type)

        ls_feature.append(extract_features(ls_fileNames[iter_k],model_archType))  
    print("---get_img2vec--len(ls_feature----",len(ls_feature))
    print("----get_img2vec-len(ls_feature[0]----",len(ls_feature[0]))
    
    pickle.dump(ls_feature, open('./pickle_files/'+str(str_cnst_caltech)+'_img2vec_.pickle', 'wb'))
    print("----get_img2vec---Pickle File Written--->>\n",str(str_cnst_caltech)+'_img2vec_.pickle')
    return ls_feature

# def read_ftrs_pickle(str_cnst_caltech):
#     # Reading from Pickled Files 
#     ls_ftrs = pickle.load(open('./pickle_files/'+str(str_cnst_caltech)+'_img2vec_.pickle','rb')) 
#     print("----read_ftrs_pickle-len(ls_feature----",len(ls_ftrs))
#     print("----read_ftrs_pickle-len(ls_feature[0]----",len(ls_ftrs[0]))
#     return ls_ftrs

# def read_fName_pickle(str_cnst_caltech):
#     # Reading from Pickled Files 
#     ls_fileNames = pickle.load(open('./pickle_files/'+str(str_cnst_caltech)+'_ls_fileNames_.pickle','rb'))
#     print("----Type(filenames---",type(ls_fileNames)) #LIST -validation- OK     
#     return ls_fileNames
    

if __name__ == "__main__":
    model_archType = 'resnet'
    model_archType = model_picker(model_archType)

    from datetime import datetime
    dt_time_now = datetime.now()
    minute_now = dt_time_now.strftime("_%m_%d_%Y_%H_%M")
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

    # input data path -- cropped_images
    root_dir_1 = "./datasets/input_dir/imgs/knn_imgs/faces_1/" # TODO
    
    root_dir_label_name = root_dir_1.split('/')[-2] 
    print("----root_dir_label_name-",root_dir_label_name)
    str_cnst_caltech = "caltech_nn_"+str(root_dir_label_name)#+str(minute_now) #
    # TODO -- Fails as next Min Read Pickle has new name 

    
    # TODO -- toggle below code 
    ls_fileNames = get_fileNames_pickle(root_dir_1,str_cnst_caltech)
    print("---main---ls_fileNames----",len(ls_fileNames))
    ls_ftrs = get_img2vec(ls_fileNames,str_cnst_caltech,model_archType,root_dir_label_name)

    # # TODO -- toggle below code 
    # ls_fileNames = read_fName_pickle(str_cnst_caltech)
    # ls_ftrs = read_ftrs_pickle(str_cnst_caltech)
    # neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='euclidean').fit(ls_ftrs)
    # ls_similar_image_paths ,ls_distances  = query_img_neighbors(ls_ftrs,ls_fileNames,neighbors,root_dir_label_name)
    # print("----LEN(--ls_similar_image_paths----------",len(ls_similar_image_paths))

    # meta_data_df = plot_nn_images(ls_similar_image_paths ,ls_distances,root_dir_label_name)

    # file_extn_type = ".png"
    # get_all_files_df(root_dir_1,root_dir_label_name,file_extn_type)

    # #TODO -- NotRequired>> wrapper_plot_nn_images(ls_ftrs,ls_fileNames,neighbors,root_dir_label_name)
