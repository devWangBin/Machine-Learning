

hand_Image = Hand_Gesture_Image;
numImg = size(hand_Image,1);
[~,m,n,~] = size(hand_Image);
hand_Image_gray = uint8(zeros(numImg,m,n));

for i=1:numImg
    img = squeeze(hand_Image(i,:,:,:));
    grayImg = rgb2gray(img);
    
    hand_Image_gray(i,:,:) = uint8(grayImg);
end

save('hand_Image_gray.mat','hand_Image_gray');