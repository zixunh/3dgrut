python prepare_scannetpp.py \
    --path /media/scannetpp/demo/0a5c013435/dslr/ \
    --src resized_images \
    --dst image_undistorted_fisheye 


python prepare_scannetpp_fish2equi.py \
    --path /media/scannetpp/demo/0a5c013435/dslr/ \
    --src resized_images \
    --dst images_equidist