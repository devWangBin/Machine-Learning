#coding:utf-8

import tensorflow as tf
import numpy as np
import gesture_forward
import gesture_backward
from image_processing import func5,func6
import cv2

def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        
        x = tf.placeholder(tf.float32,[
            1,
            gesture_forward.IMAGE_SIZE,
            gesture_forward.IMAGE_SIZE,
            gesture_forward.NUM_CHANNELS])    
        #y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
        y = gesture_forward.forward(x,False,None)
        
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(gesture_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(gesture_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] 
                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1



def application01():
    testNum = input("input the number of test pictures:")
    testNum = int(testNum)
    for i in range(testNum):
        testPic = input("the path of test picture:")
        img = func5(testPic)
        cv2.imwrite(str(i)+'ttt.jpg',img)   
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        img = img.reshape([1,100,100,1])
        img = img.astype(np.float32)
        img = np.multiply(img, 1.0/255.0)
#        print(img.shape)
#        print(type(img))        
        preValue = restore_model(img)
        print ("The prediction number is:", preValue)

def application02():
    
    #vc = cv2.VideoCapture('testVideo.mp4')
    vc = cv2.VideoCapture(0)
    # 设置每秒传输帧数
    fps = vc.get(cv2.CAP_PROP_FPS)
    # 获取视频的大小
    size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 生成一个空的视频文件
    # 视频编码类型
    # cv2.VideoWriter_fourcc('X','V','I','D') MPEG-4 编码类型
    # cv2.VideoWriter_fourcc('I','4','2','0') YUY编码类型
    # cv2.VideoWriter_fourcc('P','I','M','I') MPEG-1 编码类型
    # cv2.VideoWriter_fourcc('T','H','E','O') Ogg Vorbis类型，文件名为.ogv
    # cv2.VideoWriter_fourcc('F','L','V','1') Flask视频，文件名为.flv
    #vw = cv2.VideoWriter('ges_pro.avi',cv2.VideoWriter_fourcc('X','V','I','D'), fps, size)
    # 读取视频第一帧的内容
    success, frame = vc.read()
#    rows = frame.shape[0]    
#    cols = frame.shape[1]
#    t1 = int((cols-rows)/2)
#    t2 = int(cols-t1)
#    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
#    frame = cv2.warpAffine(frame,M,(cols,rows))
#    frame = frame[0:rows, t1:t2]
#    cv2.imshow('sd',frame)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    while success:
        
        #90度旋转        
#        img = cv2.warpAffine(frame,M,(cols,rows))
#        img = img[0:rows, t1:t2]
        img = func6(frame)
        img = img.reshape([1,100,100,1])
        img = img.astype(np.float32)
        img = np.multiply(img, 1.0/255.0)
        preValue = restore_model(img)
        # 写入视频
        cv2.putText(frame,"Gesture:"+str(preValue),(50,50),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),1)
        #vw.write(frame)
        cv2.imshow('gesture',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 读取视频下一帧的内容
        success, frame = vc.read()
    
    vc.release()
    cv2.destroyAllWindows()    
    print('viedo app over!')


def main():
    #application01()
    application02()
    
if __name__ == '__main__':
	main()		
