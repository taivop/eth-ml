% I = rgb2gray(imread('image_0058.jpg'));
bin = 8;
angle = 360;
L=3;
% roi = [1;225;1;300];
% p = anna_phog(I,bin,angle,L,roi)


pure_png = rgb2gray(imread('/Users/davidtedaldi/Desktop/spots_A_8_1_top_left_nuclID_1_p1_label_2_p2_label_2_stained_0 (1).tif'));
roi = [1;80;1;80];
p = anna_phog(pure_png,bin,angle,L,roi)
size(p)
