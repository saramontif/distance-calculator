from __future__ import print_function, division
import pandas as pd
from meye import MEImage
from scipy.ndimage.filters import maximum_filter
import matplotlib.patches as patches
from scipy import signal as sg

%pylab inline
plt.rcParams['image.cmap'] = 'gray'


def find_corners(img, rects):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    sobel_kernel_y = numpy.rot90(sobel_kernel_x)
    my_image = np.arange(8) + np.arange(8)[:,np.newaxis] 
    new_image = img
    
    Ix = sg.convolve2d(new_image, sobel_kernel_x, "same") 
    Iy = sg.convolve2d(new_image, sobel_kernel_y, "same") 
    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy
    
    G = np.ones((7,7))  / 49
    Sxx = sg.convolve2d(Ixx, G, "same")
    Syy = sg.convolve2d(Iyy, G, "same")
    Sxy= sg.convolve2d(Ixy, G, "same")
    
    R = ((Sxx*Syy) - (Sxy**2)) / (Sxx+Syy+0.1)
    
    max_num = maximum_filter(R, 15)
    mask = (R == max_num)
   
    return mask;


def index_by_mask(mask):
    y,x = mask.shape
    list_index = list(zip(np.where(mask == True)[0], np.where(mask == True)[1]))
    return list_index


# def print_points(list_points):
#     i = np.copy(img_prev)
#     i[mask_prev] = 255

#     imshow(i, cmap='gray',origin='lower')
    
    
    
def match(mask1, mask2):
    y_n,x_n = (mask2.shape[1] / mask1.shape[1], mask2.shape[0] / mask1.shape[0],)
    indexes1, indexes2 = index_by_mask(mask1), index_by_mask(mask2)
    result = []
    for point1 in indexes1:
        for point2 in indexes2:
            p = (point1[0] * x_n, point1[1] * y_n)
            if abs(p[0] - point2[0]) < 2 and abs(p[1] - point2[1]) < 2:
                result.append([(int(point1[0]),int(point1[1])),((point2)[0],(point2)[1])])
    return (np.array(result))

def plot_match(list_points, img_prev, img_curr):
    
    for i in range(3):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        fig2 = plt.figure()
        ax2= fig2.add_subplot(111)

        for pair in list_points:
            ax1.imshow(img_prev, origin="lower")
            ax1.scatter(pair[0][0], pair[0][1])

            plt.imshow(img_curr, origin="lower")
            ax2.scatter(pair[1][0], pair[1][1])

def find_distance(df, i,points, im_p, im_c):
    egomotion = df.iloc[i].egoMotion

    focal = df.iloc[i].focal
    R = egomotion[:3,:3]
    T = egomotion[:-1,-1]

    x_prev, y_prev = points[0][0]
    x_curr, y_curr = points[0][1]

    ex = focal*(T[0]/T[2])

    origin_curr = im_c.origin
    x_curr -= origin_curr[0]
    y_curr -= origin_curr[1]
    origin_prev = im_p.origin
    x_prev -= origin_prev[0]
    y_prev -= origin_prev[1]

    prev = [x_prev, y_prev, focal]
    xrot, yrot, frot = np.matmul(R, prev)
    Z = T[2]*((xrot-ex)/(xrot-x_curr))
    return Z
            
df = pd.read_pickle("store.pickle")      
for i in range(3):
    img_prev, rect_prev = MEImage.from_file(df.iloc[i].prevImage), df.iloc[i].prevRect
    im_p = img_prev
    img_prev = img_prev.im[rect_prev[2]:rect_prev[3],rect_prev[0]:rect_prev[1]]
    img_curr, rect_curr = MEImage.from_file(df.iloc[i].currImage), df.iloc[i].currRect
    im_c = img_curr
    img_curr = img_curr.im[rect_curr[2]:rect_curr[3],rect_curr[0]:rect_curr[1]]

    # img_prev, rect_prev = df.iloc[2].prevImage, df.iloc[2].prevRect
    mask_prev = find_corners(img_prev, rect_prev)
    # img_curr, rect_curr = df.iloc[2].currImage, df.iloc[2].currRect
    mask_curr = find_corners(img_curr, rect_curr)

    x_n, y_n = (mask_prev.shape[1] / mask_curr.shape[1], mask_prev.shape[0] / mask_curr.shape[0])
    r = np.array([y_n,x_n])
    indexes1 = index_by_mask(mask_prev)
    indexes1 = indexes1 * r

    index_by_mask(mask_curr)

    points = match(mask_prev, mask_curr)

    points[:,0][:,0]+=rect_prev[0]
    points[:,0][:,1]+=rect_prev[2]
    points[:,1][:,0]+=rect_curr[0]
    points[:,1][:,1]+=rect_curr[2]

    print(find_distance(df, i,points, im_p, im_c))
           

