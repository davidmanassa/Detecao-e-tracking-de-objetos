# Pacotes importados
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from random import randint

# Ditância para considerar que é o mesmo objeto
max_distance_same_object = 20
# Tempo para considerar objeto perdido em mili segundos
time_to_lose_object = 30000
# Obriga a deteção a executar de x em x frames
frames_to_detect = 40

# Classe VideoStream para lidar com o streaming de video através de uma webcam em um processo diferente
class VideoStream:
    def __init__(self, resolution=(640,480), framerate=30):
        # Inicializa a câmara e a stream de imagens
        # 0 corresponde à câmara 0 em uma lista de câmeras conectadas ao Raspberry Pi. Se apenas uma câmera conectada, selécionará sempre essa.
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
            
        # Lê o primeiro frame da stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variável para controlar a câmara quando esta for parada
        self.stopped = False

    def start(self):
        # Inicia o Thread que irá ler frames do stream de video
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Mantêm um loop infinito até o Thread ser parado
        while True:
            # Se a câmara parar, para o Thread
            if self.stopped:
                # Fecha os recursos da câmara
                self.stream.release()
                return

            # Obtêm o frame seguinte da stream de video
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Retorna o frame mais recente
        return self.frame

    def stop(self):
        # Indica que a câmara e o Thread devem parar
        self.stopped = True

# Define e analisa argumentos de entrada
modeldir="tfmodel"
graph='detect.tflite'
labels='labelmap.txt'
threshold=0.5
resolution='640x480'

MODEL_NAME = modeldir
GRAPH_NAME = graph
LABELMAP_NAME = labels
min_conf_threshold = float(threshold)
resW, resH = resolution.split('x')
imW, imH = int(resW), int(resH)

# Importação das bibliotecas do TensorFlow
# Se tflite_runtime estiver instalado, importa de tflite_runtime, se não importa do TensorFlow normal
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Obtêm o caminho até a pasta atual
CWD_PATH = os.getcwd()

# Caminho para o ficheiro .tflite, que contêm o modelo que irá ser usado para a deteção de objetos.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Caminho para o ficheiro mapa de etiquetas (label map file)
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Carrega o mapa de etiquetas
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Pequeno arranjo, quando o mapa de etiquetas está a usar o COCO "starter model" do website oficial:
# https://www.tensorflow.org/lite/models/object_detection/overview
# A primeira etiqueta é '???', que terá de ser removida
if labels[0] == '???':
    del(labels[0])

# Carrega o modelo do TensorFlow Lite.
interpreter = Interpreter(model_path=PATH_TO_CKPT)

# Alocação de mais Threads para o trabalho do TensorFlow
interpreter.set_num_threads(2)
interpreter.allocate_tensors()

# Obtêm detalhes do modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Inicia o cálculo para o numero de FPS
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Inicializa a stream de video
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1) # 1 Segundo // Tempo para a câmara inicializar e começar a capturar


# Inicialização do módulo MultiTracker presente na biblioteca OpenCV
multiTracker = cv2.MultiTracker_create()
# Booleano responsável por indicar quando irá se realizar uma deteção de objetos invês do traking
run_detector = True
# Lista de caixas e cores (atualiza quando é executado o detetor)
bboxes = [] 
colors = []
predictions = [] # Lista de tupulos: (id, object, percentage)
last_id = 0 # Variável auxiliar

ids_checked = []
# Lista de rastreamentos falhados, para que possam ser recuperados
# Composta por tuplos (tuplo caixa, tuplo cor, tuplo predicção, tempo em milissegundos de quando falhou)
failed_detections = list([])


frame_counter = 0
fail_counter = 0

# Variável auxilar, apenas usada para debuging
state_change = True

cv2.setNumThreads(1)

num_fps = []
def avgfps():
    sum = 0
    for fps in num_fps:
        sum += fps
    return sum/len(num_fps)

def getOldObject(new_box, object_name):
    # Retorna um antigo objeto se o centroide da caixa e nome de objeto coincidirem

    global failed_detections
    
    new_failed_detections = failed_detections.copy()
    element_to_return = None
    removed = 0

    atual_time = int(round(time.time() * 1000))
    
    for i, obj in enumerate(failed_detections):
        """
        0 Caixa delimitadora
        1 Cor
        2 Predicção
        3 Tempo de quando deixou de ser detectado
        """
        
        # Vamos verificar o tempo dos rastreamentos falhados (menos interacções na próxima vez)
        if (obj[3] + time_to_lose_object < atual_time):
            # Já passou 30 segundos!
            # Vamos remover da lista
            del new_failed_detections[i-removed]
            removed += 1
            print("Perdemos o objeto [" + str(obj[2][0]) + "] " + obj[2][1] + "\n Tempo passado: " + str((atual_time - obj[3])))
            continue
        else:
            # Ainda não passou 30 segundos!
            
            # É o mesmo objeto ?    
            if obj[2][1] == object_name:
                
                box1 = obj[0]
                
                box_centroid = (int(new_box[0]+(new_box[2]/2)), int(new_box[1]+(new_box[3]/2)))
                box1_centroid = (int(box1[0]+(box1[2]/2)), int(box1[1]+(box1[3]/2)))
        
                # Vamos comparar os centroids com uma distância de max_distance_same_object pixeis
                if box_centroid[0] - box1_centroid[0] < max_distance_same_object and box_centroid[0] - box1_centroid[0] > -max_distance_same_object:
                    if box_centroid[1] - box1_centroid[1] < max_distance_same_object and box_centroid[1] - box1_centroid[1] > -max_distance_same_object:
                        
                        # Os centroides estão próximos!
                        
                        # Vamos prever que seja o mesmo objeto, retornar:
                        element_to_return = obj
                        
                        # Como vamos retornar, removemos também da lista!
                        del new_failed_detections[i-removed]
                        
                        # Já temos o que queriamos. Vamos embora! Analisamos os restantes tempos superiores a 30 segundos para a próxima.
                        break
                        
    failed_detections = new_failed_detections
    
    return element_to_return

def getData(box, object_name):
    # Veirifica através de Centroid Traking se um objeto existe
    # Retorna i na lista
    
    for i, box1 in enumerate(bboxes):
        
        if object_name == predictions[i][1]:
        
            box_centroid = (int(box[0]+(box[2]/2)), int(box[1]+(box[3]/2)))
            box1_centroid = (int(box1[0]+(box1[2]/2)), int(box1[1]+(box1[3]/2)))
            
            # max_distance_same_object pixeis
            if box_centroid[0] - box1_centroid[0] < max_distance_same_object and box_centroid[0] - box1_centroid[0] > -max_distance_same_object:
                if box_centroid[1] - box1_centroid[1] < max_distance_same_object and box_centroid[1] - box1_centroid[1] > -max_distance_same_object:            
                   
                    # É o mesmo
                    object_id = predictions[i][0]
                    ids_checked.append(object_id)
                    
                    # Retorna i na lista
                    return i
            
    return -1
   
def createTraker(frame):
    for bbox in bboxes:
        multiTracker.add(cv2.TrackerMOSSE_create(), frame, bbox) # MOSSE Tracker

while True:
    
    # Forçamos o detector a executar a cada frames_to_detect frames
    frame_counter += 1
    if (frame_counter > frames_to_detect):
        run_detector = True
        state_change = True
        frame_counter = 0

    # Inicia timer, para cálculo de FPS
    t1 = cv2.getTickCount()

    # Obtemos o frame da stream de video
    frame = videostream.read()

    # Se o frame for escolhido para correr o detetor de objetos
    if run_detector:
        
        # Deteção de objetos
    
        # Apenas um Log informativo para a consola
        if state_change:
            print ("Detecting")
            state_change = False
            
        # Novas listas que irão servir de auxilio para repor as anteriores
        new_bboxes = []
        new_colors = []
        new_predictions = []
        checked = []
        
        # Efetuamos alguns ajustes à imagem para poder ser intrepertada pelo TensorFlow
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        # Executa a deteção, correndo o modelo com a imagem (frame) de input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Recebe os resultados da deteção
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Coordenadas das caixas delimitadoras do objetos encontrados
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Index das classes dos objetos detetados
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confiança dos objetos detetados
        # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Numero total de objetos detetados (Impreciso e não preciso)

        # Corre por todas as deteções
        for i in range(len(scores)):
            
            # Verifica se a percentagem de certeza é maior que a minima indicada
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Pega as coordenadas da caixa delimitadora
                # O interpretador pode devolver coordenadas fora das dimenções da imagem, vamos força las a pertencerem à imagem usando o max() e min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                object_name = labels[int(classes[i])]
                
                # Vamos verificar se este já existia
                # Posição de objeto na lista (Centroid Traking)
                original_i = getData((xmin, ymin, xmax-xmin, ymax-ymin), object_name)
                
                # Atualizamos o ID do ultimo objeto
                last_id += 1
                
                new_box = (xmin, ymin, xmax-xmin, ymax-ymin)
                new_color = (randint(64, 255), randint(64, 255), randint(64, 255))
                # Atribui novo id, porêm poderá ser reposto por um já existente
                new_predict = (last_id, object_name, int(scores[i]*100)) ## Do nothing
                     
                # Objeto já existe
                if original_i != -1:
                    
                    last_id -= 1
                        
                    if original_i not in checked:
                        
                        new_color = colors[original_i]
                        new_predict = (predictions[original_i][0], predictions[original_i][1], int(scores[i]*100)) # Atualiza percentagem

                        checked.append(original_i)
                    
                    else:
                        
                        # Objeto existe, porêm já está a ser usado
                        # Provavelmente existem 2 objetos do mesmo tipo no perto um do outro
                        continue
                                        
                else:
                    # Vamos verificar se existem objeto perdidos anteriormente

                    old_obj = getOldObject(new_box, object_name)
                    
                    if (old_obj != None):

                        new_color = old_obj[1]
                        new_predict = (old_obj[2][0], old_obj[2][1], int(scores[i]*100)) # Atualiza percentagem
                        last_id -= 1
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), new_color, 2)

                # Desenha caixa 
                label = '[%d] %s: %d%%' % (new_predict[0], new_predict[1], int(scores[i]*100)) # Example: '[0] person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0]+10, label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
                # Incrementa listas
                new_bboxes.append(new_box)
                new_colors.append(new_color)
                new_predictions.append(new_predict)
        
        if (len(new_predictions) > 0):

            run_detector = False
            state_change = True
            
            # Guarda deteções falhadas
            for i, predict in enumerate(predictions):
                if predict[0] not in ids_checked:
                    failed_detections.append((bboxes[i], colors[i], predictions[i], int(round(time.time() * 1000))))

            # Repoe lista para a próxima vez        
            ids_checked = []
            
            # Repoe listas
            bboxes = new_bboxes
            colors = new_colors
            predictions = new_predictions
            
            # Inicializa rastreamento
            multiTracker = cv2.MultiTracker_create()
            createTraker(frame)
        
    else:
        # Rastreamento de objetos
        if state_change:
            print ("Traking")
            state_change = False
        
        # Obtem localização atualizada dos objetos nos restantes frames, através do tracker
        success, boxes = multiTracker.update(frame)
        
        if not success:
            print("Fail no rastreador")
            run_detector = True
            state_change = True
     
        # Draw tracked objects
        for i, newbox in enumerate(boxes):
            
            xmin = int(newbox[0])
            ymin = int(newbox[1])
            xmax = int(newbox[0] + newbox[2])
            ymax = int(newbox[1] + newbox[3])
            # Need to check if overrides screen ?? I think no in this way
        
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), colors[i], 2)
               
            prediction = predictions[i]
            
            # Desenha caixa
            label = '[%d] %s: %d%%' % (prediction[0], prediction[1], prediction[2]) # Example: '[0] person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0]+10, label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0]+10, label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Guarda ultima posição da caixa
            bboxes[i] = newbox
            
    # Desenha FPS
    #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),2,cv2.LINE_AA)
    num_fps.append(frame_rate_calc)

    # Todos os resultados foram desenhados no ecrã, tempo de os mostrar.
    cv2.imshow('Object detector', frame)

    # Calculo de FPS
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Pressionar 'q' para sair
    if cv2.waitKey(1) == ord('q'):
        break

print("Média de FPS: " + str(avgfps()))

# Limpar data
cv2.destroyAllWindows()
videostream.stop()
