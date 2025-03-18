#copied from fast_coref_try1.py

from fastcoref import LingMessCoref
from fastcoref import FCoref
from tqdm import tqdm
from pathlib import Path

import jellyfish
import spacy
import os
import pickle

#cross document records
cd_record_ent_id_list=[]
cd_record_syn_list_list=[]
cd_record_sent_list=[]
cd_record_count_list=[]

model = LingMessCoref(device='cuda:0')#coref model
nlp = spacy.load("en_core_web_sm")#spacy model


def abbreviation_similarity_score_jw(abbreviation, full_form):
    """
    Calculate the Jaro-Winkler similarity score between an abbreviation and its full form.

    Parameters:
    - abbreviation: The abbreviation to calculate the similarity for.
    - full_form: The full form against which to calculate the similarity.

    Returns:
    - Jaro-Winkler similarity score.
    """
    abbreviation = abbreviation.lower()
    full_form = full_form.lower()

    # Remove non-alphabetic characters###############
    #abbreviation = ''.join(char for char in abbreviation if char.isalpha())#############
    #full_form = ''.join(char for char in full_form if char.isalpha())####################

    # Calculate Jaro-Winkler similarity score
    similarity_score = jellyfish.jaro_winkler_similarity(abbreviation, full_form)

    return similarity_score



def abbreviation_similarity_score_ld(abbreviation, full_form):
    """
    Calculate the Levenshtein distance between an abbreviation and its full form.

    Parameters:
    - abbreviation: The abbreviation to calculate the similarity for.
    - full_form: The full form against which to calculate the similarity.

    Returns:
    - Normalized Levenshtein distance (similarity score).
    """
    abbreviation = abbreviation.lower()
    full_form = full_form.lower()

    # Remove non-alphabetic characters#####################
    #abbreviation = ''.join(char for char in abbreviation if char.isalpha())#################
    #full_form = ''.join(char for char in full_form if char.isalpha())######################

    # Calculate Levenshtein distance
    levenshtein_distance = jellyfish.levenshtein_distance(abbreviation, full_form)

    # Calculate normalized similarity score (1 - normalized distance)
    max_len = max(len(abbreviation), len(full_form))
    similarity_score = 1 - levenshtein_distance / max_len

    return similarity_score




def string_relationship(str1, str2_list):
    for str2 in str2_list:
        if str1 == str2:
            return f"'{str1}' is the same as '{str2}'"
        elif str1 in str2:
            return f"'{str1}' is part of '{str2}'"
        elif str2 in str1:
            return f"'{str2}' is part of '{str1}'"

def string_relationship_partof(str1, str2_list):
    for str2 in str2_list:
        if str1 == str2:
            return f"'{str1}' is the same as '{str2}'"
        elif str1 in str2:
            return f"'{str1}' is part of '{str2}'"

def string_relationship_exact(str1, str2_list):
    for str2 in str2_list:
        if str1 == str2.strip():
            return f"'{str1}' is the same as '{str2}'"

def string_relationship_similarity_jw(str1, str2_list):
    max = 0
    for str2 in str2_list:
        sim_score = abbreviation_similarity_score_jw(str1, str2.strip())
        if sim_score > max:
            max = sim_score
            if max>=0.90:
                return max
    return max

def string_relationship_similarity_ld(str1, str2_list):
    max = 0
    for str2 in str2_list:
        sim_score = abbreviation_similarity_score_ld(str1, str2.strip())
        if sim_score > max:
            max = sim_score
            if max>=0.75:
                return max
    return max

def ends_with(string1, string2_list):
    for string2 in string2_list:
        if len(list(string2)) <= 2 or len(list(string1)) <=2:
            continue
        if string1.lower().endswith(string2.lower()) or string2.lower().endswith(string1.lower()):
            return 1

def check_abbreviation(str1, str2_list):
  for str2 in str2_list:
    if len(list(str2)) == 1 or len(list(str1)) ==1:
      continue
    str2_ = ''.join([s[0] for s in str2.strip().lower().split()])#abbreviation of s2
    str1_ = ''.join([s[0] for s in str1.strip().lower().split()])#abbreviation of s1
    if str1.lower() == str2_.lower() or str1_.lower() == str2.lower():
      return 1



#file_names = []
#folder_path = '/home/alapan/test_files/demonetization/new_test_files'
#file_names = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#dir_path = '/home/alapan/test_files/demonetization'
#folder_paths = [f.path for f in os.scandir(dir_path) if f.is_dir()]#enlist all the media houses
cd_record_count_list = [0]*len(cd_record_count_list)
#for folder_path in folder_paths: #for each media house
folder_path = "/home/alapan/test_files/demonetization/outlookindia"
file_names = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files]#enlist all the articles published by the media house
#cd_record_count_list = [0]*len(cd_record_count_list)
no_unprocessed_files = 0
for file_name in tqdm(file_names):
    
    try:

        #print(file_name)
        f = open(file_name, "r")
        print(file_name)

        text = f.read()
        #model = LingMessCoref(device='cuda:0')
        #model = FCoref(device='cuda:0')


        preds = model.predict(texts= text)

        #print(preds)
        # Load the English model
        #nlp = spacy.load("en_core_web_sm")

        # Your text containing entities
        #text = "Apple is looking at buying U.K. startup for $1 billion. Google, headquartered in Mountain View, California, is a technology company."

        # Process the text with spaCy
        doc = nlp(text)

        # Extract entities and their types along with corresponding sentences and indices
        entity_data = {}
        sentence_text=[]
        sentence_start_indx=[]
        sentence_end_indx=[]
        for sent in doc.sents:
            sentence_text.append(sent.text)
            sentence_start_indx.append(sent.start_char)
            sentence_end_indx.append(sent.end_char)
            for entity in sent.ents:
                entity_data.setdefault(entity.text, []).append((entity.label_, sent.text, sent.start_char, sent.end_char))

        # Print entity phrases, types, corresponding sentences, and character indices
        record_ephrase=[]
        recored_etype=[]
        record_sent=[]
        record_char_start_idx=[]
        record_char_end_idx=[]

        for entity, data in entity_data.items():
            for entity_type, sentence, start_char, end_char in data:
                if entity_type == 'PERSON' or entity_type == 'ORG' or entity_type == 'GPE':
                    record_ephrase.append(entity)#entity list
                    #record_etype.append(entity_type)
                    record_sent.append(sentence)#sentence list
                    record_char_start_idx.append(start_char)#start character list
                    record_char_end_idx.append(end_char)#end character list
                    #print(f"Entity: {entity}, Type: {entity_type}")
                    #print(f"Sentence: {sentence}")
                    #print(f"Character Indices: From {start_char} to {end_char}\n")


        cluster_index_list = preds.get_clusters(as_strings=False)
        cluster_list = preds.get_clusters()

        #print(f"wd-coref cluster_list= {cluster_list}")########################

        #record_dictionary = {}
        #key_list = ['ent_id','syn_list','sent_list','count']
        #record_dictionary = dict.fromkeys(key_list,[])
        #print(record_dictionary)

        #within document (WD) records
        record_ent_id_list=[]
        record_syn_list_list=[]
        record_sent_list=[]
        record_count_list=[]

        i=-1
        for entity_phrase in record_ephrase:
            try:
                #print(i)
                entity_phrase = entity_phrase.strip()
                print(f'entity_phrase = {entity_phrase}')
                doc1 = nlp(entity_phrase)
                # Display the result
                if len(doc1)==1 and doc1[0].pos_ == 'PRON':
                    continue


                '''check if the entity is processed before, in other documents'''
                cd_coref_flag = 0
                for each_list_cd in cd_record_syn_list_list:
                    if string_relationship_similarity_jw(entity_phrase, each_list_cd)>=0.90 or string_relationship_similarity_ld(entity_phrase, each_list_cd)>=0.75 or string_relationship_exact(entity_phrase, each_list_cd) or check_abbreviation(entity_phrase, each_list_cd):
                        print('CD coref present')
                        print(f'The entity belongs to CD coref list = {each_list_cd}')
                        that_index = cd_record_syn_list_list.index(each_list_cd)
                        cd_coref_flag = 1
                        print(f'string_relationship_exact(entity_phrase, each_coref_list)= {string_relationship_exact(entity_phrase, each_list_cd) }')
                        print(f'string_relationship_similarity_jw(entity_phrase, each_coref_list) = {string_relationship_similarity_jw(entity_phrase, each_list_cd)}')
                        print(f'string_relationship_similarity_ld(entity_phrase, each_coref_list) = {string_relationship_similarity_ld(entity_phrase, each_list_cd)}')
                        print(f'ends_with(entity_phrase, each_coref_list) = {ends_with(entity_phrase, each_list_cd)}')
                        print(f'check_abbreviation(entity_phrase, each_coref_list) = {check_abbreviation(entity_phrase, each_list_cd)}')
                        print('\n\n')
                        break



                '''if the entity is already processed within this docuemnt then pass'''
                duplicate_flag = 0
                for each_list in record_syn_list_list:
                    if string_relationship_similarity_jw(entity_phrase, each_list)>=0.90 or string_relationship_similarity_ld(entity_phrase, each_list)>=0.75 or string_relationship_exact(entity_phrase, each_list) or check_abbreviation(entity_phrase, each_list):
                        duplicate_flag = 1
                        break

                if duplicate_flag == 1:#if this entity already processed before then continue
                    print(f'entity {entity_phrase} already processed WD \n')
                    print('\n')
                    continue

                
                i+=1#track the index of the current entity phrase (as there are more than one appearance of same entity phrase)
                flag = 0
                for each_coref_list in cluster_list:
                    
                    if string_relationship_exact(entity_phrase, each_coref_list) or string_relationship_similarity_jw(entity_phrase, each_coref_list)>=0.90 or string_relationship_similarity_ld(entity_phrase, each_coref_list)>=0.75 or check_abbreviation(entity_phrase, each_coref_list): #if the entity present in any of the WD-coref_list
                        #print(f'entity phrase = {entity_phrase}')#############################
                        print('WD coref present')  
                        print(f'the entity has WD coref_list = {each_coref_list}')##############################
                        flag = 1 # entity has WD-coreference
                        print(f'string_relationship_exact(entity_phrase, each_coref_list)= {string_relationship_exact(entity_phrase, each_coref_list) }')
                        print(f'string_relationship_similarity_jw(entity_phrase, each_coref_list) = {string_relationship_similarity_jw(entity_phrase, each_coref_list)}')
                        print(f'string_relationship_similarity_ld(entity_phrase, each_coref_list) = {string_relationship_similarity_ld(entity_phrase, each_coref_list)}')
                        print(f'ends_with(entity_phrase, each_coref_list) = {ends_with(entity_phrase, each_coref_list)}')
                        print(f'check_abbreviation(entity_phrase, each_coref_list) = {check_abbreviation(entity_phrase, each_coref_list)}')
                        print('\n\n')
                        break
                    else:
                        flag = 0 #does not have WD-coreference

                if flag == 1: 
                    #print('WD coref present')   
                    ent_pharse_to_store = entity_phrase#may be stored
                    #ent_id = len(record_ent_id_list)#to store
                    ent_id = len(record_ent_id_list)#to store
                    syn_set_list = each_coref_list#to store
                    _list_target_sentence=[] #list of sentences containing the target entity or its WD-coreference
                    syn_set_list_index = cluster_index_list[cluster_list.index(each_coref_list)]
                    #print(syn_set_list_index)
                    for each_item in syn_set_list_index:
                        ent_last_index = each_item[0]
                        for s_index in sentence_end_indx:#each sentence end index, check if it greater than the entity's last index 
                            if s_index > ent_last_index:
                                target_sentence = sentence_text[sentence_end_indx.index(s_index)]
                                _list_target_sentence.append(target_sentence)
                                break
                    sent_list = _list_target_sentence#to store
                    #print(sent_list)
                    count = 1

                    record_ent_id_list.append(ent_id)
                    syn_set_list_filtered = [original_item for original_item in syn_set_list if nlp(original_item)[0].pos_ != 'PRON']
                    record_syn_list_list.append(syn_set_list_filtered)
                    record_sent_list.append(sent_list)
                    record_count_list.append(count)

                    if cd_coref_flag == 1:#if the entity has cd-coref with previously processed entities 
                        #print('CD coref present')
                        list1 = cd_record_syn_list_list[that_index]#in cd record
                        list2 = syn_set_list_filtered#in wd record
                        list3 = list(set(list1).union(set(list2)))
                        cd_record_syn_list_list[that_index] = list3# modify CD coref_list
                    
                        list1 = cd_record_sent_list[that_index]
                        list2 = sent_list
                        list3 = list(set(list1).union(set(list2)))
                        cd_record_sent_list[that_index] = list3 # modify sentence list from CD content

                        cd_record_count_list[that_index] +=1
                    else:# if the entity does not has cd-coref with previous processed entities
                        print('CD coref not present')
                        cd_record_ent_id_list.append(len(cd_record_ent_id_list))
                        cd_record_sent_list.append(sent_list)
                        cd_record_syn_list_list.append(syn_set_list_filtered)
                        cd_record_count_list.append(1)


                    #print(i)
                    #print('WD coref present')
                    #print(record_dictionary)
                    #print(record_ent_id_list)
                    #print(record_syn_list_list)
                    #print(record_sent_list)

                    #print(f'Records after processing entity= {entity_phrase}')
                    #for printing_i in range(len(cd_record_ent_id_list)):##################################
                    #    print((cd_record_ent_id_list[printing_i], cd_record_syn_list_list[printing_i]))
                        

                    
                else:#WD coref is not present
                    #print(i)
                    print('WD coreference not present')
                    ent_id = len(record_ent_id_list)#to store
                    syn_set_list = [entity_phrase]# syn_set list contains single entity phrase, #to store
                    
                    sent_list = []
                    for sentence_item in sentence_text:
                        if  entity_phrase in sentence_item:
                            #print(sent_idx)
                            #print(sentence_text[sentence_end_indx.index(sent_idx)])
                            sent_list.append(sentence_item)
                            break
                    count = 1
                    record_ent_id_list.append(ent_id)
                    record_syn_list_list.append(syn_set_list)
                    record_sent_list.append(sent_list)
                    record_count_list.append(count)


                    if cd_coref_flag == 1:
                        #print('CD coref present')
                        list1 = cd_record_syn_list_list[that_index]
                        list2 = syn_set_list
                        list3 = list(set(list1).union(set(list2)))
                        cd_record_syn_list_list[that_index] = list3# modify CD coref_list
                    
                        list1 = cd_record_sent_list[that_index]
                        list2 = sent_list
                        list3 = list(set(list1).union(set(list2)))
                        cd_record_sent_list[that_index] = list3 # modify sentence list from CD content

                        cd_record_count_list[that_index] +=1
                    else:# if the entity does not has cd-coref with previous processed entities
                        print('CD coref not present')
                        cd_record_ent_id_list.append(len(cd_record_ent_id_list))
                        cd_record_sent_list.append(sent_list)
                        cd_record_syn_list_list.append(syn_set_list)
                        cd_record_count_list.append(1)

                    
                    #print(record_dictionary)
                    #print(record_ent_id_list)
                    #print(record_syn_list_list)
                    #print(record_sent_list)

                    #print(f'Records after processing entity= {entity_phrase}')
                    #for printing_i in range(len(cd_record_ent_id_list)):########################
                    #    print((cd_record_ent_id_list[printing_i], cd_record_syn_list_list[printing_i]))
                    #print(f'cd_record_count_list={cd_record_count_list}')
                print('\n')
            except:
                continue
        print('-------------------------------------------------------------------------------------------------------------')
        f.close()
        #break #for a specific file
    except:
        no_unprocessed_files += 1
        continue
#print(cd_record_ent_id_list)
#print(cd_record_syn_list_list)
#print(cd_record_sent_list)
#print(cd_record_count_list)

output_file = os.path.join(folder_path, 'output.txt')
f_write = open(output_file,'w+')
for print_i in range(len(cd_record_ent_id_list)):
    if cd_record_count_list[print_i]>0:
        f_write.write(str(cd_record_ent_id_list[print_i])+'\t'+str(cd_record_syn_list_list[print_i])+'\t'+str(cd_record_count_list[print_i]))
        f_write.write('\n')
f_write.close()

print(f'records of mediahouse {Path(folder_path).name} is written to output text file')

pickle_file = os.path.join(folder_path, 'output.pkl')
#f_pkl = open(pickle_file, 'wb')
# Write the lists to the pickle file
ent_id_list_pkl =[]
ent_syn_list_pkl =[]
ent_sent_list_pkl =[]
for w_i in range(len(cd_record_count_list)):
    if cd_record_count_list[w_i] > 0:
        ent_id_list_pkl.append(cd_record_ent_id_list[w_i])
        ent_syn_list_pkl.append(cd_record_syn_list_list[w_i])
        ent_sent_list_pkl.append(cd_record_sent_list[w_i])
    else:
        continue
with open(pickle_file, 'wb') as f_pkl:
    pickle.dump((ent_id_list_pkl, ent_syn_list_pkl, ent_sent_list_pkl), f_pkl)

print(f'records of mediahouse {Path(folder_path).name} is written to pkl file')

print(f'number of unprocessed files = {no_unprocessed_files}')
