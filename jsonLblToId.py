import json

Json_dict = json.load(open('dataset_training.json'))

labels = []
c = 0
# labels_per_vid = []
# l = 0
for vidName in Json_dict:
    # print (vidName)
    # for label in Json_dict[vidName]:
    #     if (label['label'] not in labels_per_vid):
    #         labels_per_vid.append(label['label'])
    #         l = l+1
    # print(l, 'labels_per_vid')
    for label in Json_dict[vidName]:
        if (label['label'] not in labels):
            labels.append(label['label'])
            c = c+1
print(c, 'labels')
# id_to_label = dict(enumerate(labels))
# print ('ID to labels: ', id_to_label )
label_to_id = dict(map(reversed, enumerate(labels)))
print('label to ID: ', label_to_id)
