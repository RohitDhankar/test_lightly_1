
## SOURCE -- code inspired by - PyImageSearch or some other sources ? 

from datetime import datetime
from tqdm import tqdm
import numpy as np , cv2 , pandas as pd , io , PIL.Image as Image , os

def get_cntrs(img_for_cntrs):#,img_path):
    """
    some basic OpenCV Code 
    getting contours from images
    
    """
    try:
        #print("---type(img_for_cntrs----",type(img_for_cntrs)) #<class 'str'>
        #img_init = cv2.imread(str(img_path)+str(img_for_cntrs)+".png") 
        img_init = cv2.imread(img_for_cntrs) 
        #print("---type(img_init----",type(img_init)) #<class 'numpy.ndarray'>
        print("-[INFO_get_cntrs]-img_init.shape--",img_init.shape)
        img_gray = cv2.cvtColor(img_init, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # find the biggest countour (c) by the area
        max_cntr_area = max(contours, key = cv2.contourArea)
        #ERROR >> max() arg is an empty sequence
        print("-[INFO_get_cntrs]--type(max_cntr_area--",type(max_cntr_area))
        #print("-[INFO_get_cntrs]--max_cntr_area--",max_cntr_area) # Dont large Print 
        #x,y,w,h = cv2.boundingRect(c)
        max_cntr_perimeter = sorted(contours, key = lambda indl_cntr : cv2.arcLength(indl_cntr , False),reverse = True) 
        #ERROR >> arcLength() missing required argument 'closed' (pos 2) == SOLVED == reverse = True
        #ERROR >> arcLength() missing required argument 'curve' (pos 1)
        print("-[INFO_get_cntrs]--type(max_cntr_perimeter---",type(max_cntr_perimeter)) ## <class 'list'> -- LIST of ndArays
        print("-[INFO_get_cntrs]--type(max_cntr_perimeter---",len(max_cntr_perimeter)) #
        #print("-[INFO_get_cntrs]--max_cntr_perimeter---",max_cntr_perimeter[0])
        return contours, hierarchy , img_init , max_cntr_area , max_cntr_perimeter

    except Exception as err_get_cntrs:
        print('-[ERROR]--err_get_cntrs---\n',err_get_cntrs) 
        if "empty" in str(err_get_cntrs): # "max() arg is an empty sequence"
            contours = "ERROR_contours"
            hierarchy = "ERROR_hierarchy"
            max_cntr_area = "ERROR_max_cntr_area"
            max_cntr_perimeter = "ERROR_max_cntr_perimeter"
            return contours, hierarchy , img_init , max_cntr_area , max_cntr_perimeter
        else:
            print('-[ERROR]--err_get_cntrs-inside-ELSE-->>\n',err_get_cntrs)
            pass


def get_dict_maxEle(array_max_ele):
    """
    """
    maxEle_res = np.amax(array_max_ele)
    print('Max ele maxEle_res ---\n', maxEle_res) ## for Blue Always -- 241
    unique, counts = np.unique(array_max_ele, return_counts=True)
    dict_unq= dict(zip(unique, counts))
    print("--dict_unq-res---",dict_unq) #TODO -- see logfile -- get_color_obj1.log
    max_key_val = max(dict_unq.items(), key = lambda k : k[1])
    print("---get_dict_maxEle-max_key_val---",max_key_val) #TODO -- see logfile -- get_color_obj_2.log

def get_hsv_img(img_bgr_hsv):
    """
    HSV -  color space (hue, saturation, value)
    HSV == HSB -- also known as HSB, for hue, saturation, brightness)
    https://en.wikipedia.org/wiki/HSL_and_HSV
    """
    #TODO -- lower_green etc etc ...
    # Convert the FRAME ColorSpace from - BGR to HSV
    img_hsv = cv2.cvtColor(img_bgr_hsv,cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    #
    lower_green = np.array([110,50,50])
    upper_green = np.array([130,255,255])
    #
    lower_brown = np.array([110,50,50])
    upper_brown = np.array([130,255,255])
    #
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    #get_dict_maxEle(mask_blue)  -- Hold need further analysis on >> max_key_val ...from >> get_dict_maxEle
    # Bitwise-AND mask and original image
    res_img_hsv_blue = cv2.bitwise_and(img_bgr_hsv,img_bgr_hsv, mask= mask_blue)
    print("---type(res----",type(res_img_hsv_blue))
    print("---type(res----",res_img_hsv_blue)


def validate_img_cntrs(contours, hierarchy , img_init):
    #TODO -- Whats the VALIDATED end result of the --- img_cntrs_type --- ?? 
    pass


def draw_cntrs(contours,img_init):
    """ 
    TODO -- Whats the VALIDATED end result of the --- img_cntrs_type --- ?? 
    TODO -- relate contours count[How many contours extracted] / contours >> AREA | Dimensions etc -- to Img_Quality of cropped images 
    """
    try:
        img_cntrs = cv2.drawContours(img_init, contours, -1, (0,255,0), 1)
        print("---img_init.shape----",img_init.shape)
        img_cntrs_type = "img_cntrs_type_a"
        return img_cntrs , img_cntrs_type
    except Exception as err_draw_cntrs:
        print('---err_draw_cntrs---\n',err_draw_cntrs)
        pass

def save_img_cntrs(img_cntrs,root_dir_label_name,query_img,img_cntrs_type):
    """
    TODO -- Whats the VALIDATED end result of the --- img_cntrs_type --- ?? 
    """
    try:
        #dt_time_now = datetime.now() 
        #hour_now = dt_time_now.strftime("_%m_%d_%Y_%H_/")
        query_img = str(query_img).split("_")[0]
        path_img_cntrs = "./output_dir/knn_similar_images/"+str(root_dir_label_name)+"/"+str(img_cntrs_type)+"/"+str(query_img)
        print("---path_img_cntrs----\n",path_img_cntrs)
        #hourly_dir = os.path.join(crop_img_dir_path)
        if not os.path.exists(path_img_cntrs):
            os.makedirs(path_img_cntrs)
        complete_img_path = str(path_img_cntrs)+"/"+str(query_img)+'img_cntrs_.png'
        cv2.imwrite(complete_img_path,img_cntrs)
        return complete_img_path 
        #TODO - save path for reporting etc 
    except Exception as err_save_img_cntrs:
        print('---err_save_img_cntrs---\n',err_save_img_cntrs)
        pass



