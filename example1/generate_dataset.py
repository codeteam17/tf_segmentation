import cv2
import numpy as np

num_of_images = 100
lines_per_image = 3

output_dir = '/Users/peric/dev/tensorflow-code/example1/data'
target_size = (240, 320, 1)
out_size = (20, 15)

for i in range(num_of_images):

    if i % 100 == 0:
        print("Generating image {}...".format(i))

    # placeholder for input image
    img = np.zeros(target_size, dtype=np.uint8)

    # placeholder for groundtruth image
    img_out = np.zeros(target_size, dtype=np.uint8)

    # draw random lines (noise)
    for j in range(lines_per_image):
        start_point = ( int(np.random.uniform(0, target_size[1])), int(np.random.uniform(0, target_size[0])) )
        end_point = ( int(np.random.uniform(0, target_size[1])), int(np.random.uniform(0, target_size[0])) )
        color = (255, 0, 255)
        cv2.line(img, start_point, end_point, color)
    
    # draw the target shape
    rec_size = int(target_size[0] * 0.4)
    start_point = ( int(np.random.uniform(0, target_size[1] - rec_size)), int(np.random.uniform(0, target_size[0]-rec_size)) ) 
    end_point = ( start_point[0] + rec_size, start_point[1] + rec_size )
    cv2.rectangle(img, start_point, end_point, color, 1)

    # paint the groundtruth
    inflate = int(rec_size * 0.1)
    start_point = (start_point[0] - inflate, start_point[1] - inflate)
    end_point = (end_point[0] + inflate, end_point[1] + inflate)
    
    cv2.rectangle(img_out, start_point, end_point, color, -1)
    img_out = cv2.resize(img_out, out_size)

    # dump the images
    cv2.imwrite(output_dir + '/' + str(i) + '.png', img)
    cv2.imwrite(output_dir + '/' + str(i) + '_out.png', img_out)

    # cv2.imshow("in", img)
    # cv2.imshow("ground", img_out)
    # cv2.waitKey(100)

