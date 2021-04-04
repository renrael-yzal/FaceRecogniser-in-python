def train_model():
    from constants import input_dim
    from model import create_model
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       horizontal_flip = True)
    #print(train_datagen)
    trainset_gen=train_datagen.flow_from_directory('dataset',
                                                   target_size=input_dim[:-1],
                                                   batch_size=32,
                                                   class_mode="categorical")
    clases=trainset_gen.class_indices
    f=open("classes_indices.csv","w+")
    f.write("Class,Label\n")
    for (v,k) in clases.items():
        f.write(str(k)+","+str(v)+"\n")
        print(k,"-",v)
    f.close()
    model=create_model()
    model.fit_generator(trainset_gen,epochs=50)
    model.save("face_Detector.h5")
    print("model saved in the present directory")
train_model()
