from modelscope import AutoModel, AutoTokenizer
import torch
import argparse
from openai import OpenAI
import os
import json
from openpyxl import load_workbook
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='VideoChat with Qwen2-7B model')
    parser.add_argument('--video_contain_path', type=str, required=True, help='Path to the folder which contain input video file')
    parser.add_argument('--ctrl_id', type=int, required=True, help='Id for video path')
    parser.add_argument('--output_dir', type=str, default='outtest1', help='Root directory for output files')
    return parser.parse_args()

def find_videos_with_ctrl_id(folder_path, ctrl_id):
    folder = Path(folder_path)
    search_str = str(ctrl_id)
    
    video_paths = []
    for mp4_file in folder.rglob("*.mp4"):
        if "totalcounter("+search_str+")" in str(mp4_file):
            video_paths.append(str(mp4_file.resolve()))
    return video_paths
def get_task_name_detail_id(folder_path,ctrl_id):
    folder = Path(folder_path)
    search_str = str(ctrl_id)
    
    for mp4_file in folder.rglob("*.mp4"):
        if "totalcounter("+search_str+")" in str(mp4_file):
            video_path=str(mp4_file.resolve())
    pos1=video_path.index("<")
    pos2=video_path.index(">")
    len_path=video_path.__len__
    detail_id=int(video_path[-1])
    task_name=video_path[pos1+1:pos2]
    return detail_id,task_name
def get_video_words(folder_path,ctrl_id):
    folder=Path(folder_path+"/instructions")
    search_str=str(ctrl_id)
    json_path=""
    for json_file in folder.rglob("*.json"):
        if "totalcounter"+search_str+".json" in str(json_file):
            json_path=str(json_file.resolve())

    with open(json_path,'r')as open_json:
        load_dict=json.load(open_json)
        data1=load_dict["seen"]
        data1=data1[:3]#get three instructions as the brief description
    out_str=""
    for _ in data1:
        out_str+=_
    return out_str

def get_one_instruction(folder_path,ctrl_id):
    folder=Path(folder_path+"/instructions")
    search_str=str(ctrl_id)
    json_path=""
    for json_file in folder.rglob("*.json"):
        if "totalcounter"+search_str+".json" in str(json_file):
            json_path=str(json_file.resolve())

    with open(json_path,'r')as open_json:
        load_dict=json.load(open_json)
        data1=load_dict["seen"]
        data1=data1[:1]#get one instruction as the brief description
    out_str=""
    for _ in data1:
        out_str+=_
    return out_str#return the first instruction



def full_procedure():
    args = parse_args()
    
    model_path = os.path.abspath('checkpoint/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448')#needs to be changed to your local path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
    
    image_processor = model.get_vision_tower().image_processor
    model.config.mm_llm_compress = False  
    generation_config = {
        'do_sample': False,
        'temperature': 0.0,
        'max_new_tokens': 1024,
        'top_p': 0.1,
        'num_beams': 1
    }
    print("Beginning video captioning")
    question1 = "Describe the video in detail:"
    video_path_=find_videos_with_ctrl_id(args.video_contain_path,args.ctrl_id)
    video_path_=video_path_[0]
    output1, _ = model.chat(
        video_path=video_path_,
        tokenizer=tokenizer,
        user_prompt=question1,
        return_history=True,
        max_num_frames=512,  
        generation_config=generation_config
    )
    print("\nFirst response:")
    print(output1)
    detail_id,task_description=get_task_name_detail_id(args.video_contain_path,args.ctrl_id)
    model_name = "Qwen/Qwen3-8B"
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    true_scene=get_video_words(args.video_contain_path,args.ctrl_id)
    print(f"true scene got from json file:{true_scene}")
    first_instruction=get_one_instruction(args.video_contain_path,args.ctrl_id)

    # guide prompt
    new_prompt = f"""You are given a video caption describing a robot manipulation scene and a brief origin video description. Your task is to generate a new video caption for a text-to-video generation model.The new captions should:
* Only change the table surface like changing material  to create  differences .
* Output style is expected to look like the following strictly:The video is a demonstration of robotic manipulation, likely in a laboratory or industrial setting. It features a single robotic arm interacting with a plastic bottle. The setting is a room with a polished stainless steel countertop, which reflects overhead lights and provides a sterile, metallic backdrop for the activity. The robotic arm, marked 'AGILE X', is positioned above the bottle, which is filled with a dark liquid.At the beginning, the bottle is standing upright on the counter. The robotic arm approaches the bottle, its gripper maneuvering with precision as it positions itself. The arm's gripper then grasps the bottle firmly by its neck. As the arm lifts the bottle smoothly, the liquid inside sways gently. The entire process highlights the precision and control of the robotic arm. The camera remains static throughout, focusing on the interaction between the robotic arm and the bottle, allowing viewers to observe the detailed movements involved in the task. 
* The central focus of the  caption should be on a table's surface.The table should be made of wood, stainless steel, or marble .
* Output is expected to be brief and easy enough for diffusion model to understand.
* The final output should contain only the  new caption with no additional commentary or explanation.The video caption is: {output1} 
* The brief origin video description  is :{true_scene} ,make sure that the output content has the same meaning and  object name as the given brief description"""

    messages = [
        {"role": "user", "content": new_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    detailed_scene = content
    print("Generated scenes:")
    print(detailed_scene)
    output_base = os.path.abspath(args.output_dir)
    example_dir = os.path.join(output_base, f"total_{args.ctrl_id}")
    os.makedirs(example_dir, exist_ok=True)
    outpath=os.path.join(example_dir,"prompt.txt")
    with open (outpath,'w',encoding='utf-8') as file:
        file.write(detailed_scene)
    print(f"Successfully saved prompts to: {outpath}")
    out_control_json=os.path.join(example_dir,"control.json")
    with open(out_control_json,'w',encoding='utf-8') as json1:
        json.dump({"name": "robot_depth","prompt_path": outpath,"video_path":video_path_,"guidance": 3, "depth": {"control_path":None,"control_weight": 1.0}},json1,ensure_ascii=False)
    print(f"Successfully saved control json to: {out_control_json}")
    out_data_json=os.path.join(example_dir,"out_data.json")
    with open(out_data_json,'w',encoding='utf-8') as json2:
        json.dump({"task_name":task_description,"true_scene_from_instructions":true_scene,"first_instruction":first_instruction,"detail_task_id":detail_id,"video_origin_description":output1},json2,ensure_ascii=False)
    print(f"Successfully saved data json to: {out_data_json}")
    return True
if __name__ == "__main__":
    full_procedure()