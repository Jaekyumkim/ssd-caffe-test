from xml.dom.minidom import Document
import cv2, os

def generate_xml(name,split_lines,img_size,class_ind):
    doc = Document()  

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    annotation.appendChild(title)

    img_name=name+'.png'

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The KITTI Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)



    for split_line in split_lines:
        line=split_line.strip().split()
        if line[0] in class_ind:
            object = doc.createElement('object')
            annotation.appendChild(object)

            title = doc.createElement('name')
            title_text = doc.createTextNode(line[0])
            title.appendChild(title_text)
            object.appendChild(title)

            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(float(line[4]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(float(line[5]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(float(line[6]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(float(line[7]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)


    f = open('label/'+name+'.xml','w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()




if __name__ == '__main__':
    class_ind=('Pedestrian', 'Car', 'Cyclist')
    cur_dir=os.getcwd()
    dir=os.path.join(cur_dir,'label_2')
    train_txt= open('train.txt','w')
    test_name_size= open('test_name_size.txt','w')
    for parent, dirnames, filenames in os.walk(dir):
        for file_name in filenames:
            full_path=os.path.join(parent, file_name)
            f=open(full_path)
            split_lines = f.readlines()
            name= file_name[:-4]
            img_name=name+'.png'
            img_path=os.path.join('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/image_2',img_name)
            img_size=cv2.imread(img_path).shape
            generate_xml(name,split_lines,img_size,class_ind)
            train_txt.write('image_2/'+img_name+' '+'label/'+name+'.xml'+'\n')
            test_name_size.write(name+' '+str(img_size[0])+' '+str(img_size[1])+'\n')
