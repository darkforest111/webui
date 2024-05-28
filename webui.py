import gradio as gr

import suno_api
from comosvc.preparation_slice import process
from comosvc.infer_tool import Svc
import glob
import os
import subprocess
import torch
import soundfile
from tts_voices import SUPPORTED_LANGUAGES
import aiohttp
import json
from suno_api import *
from dotenv import load_dotenv

local_model_root="./logs"
global_model_path=None
global_config_path=None

model=None
spk_list=[]
default_config_path= "comosvc/logs/como/model_800000.pt"
default_model_path= "comosvc/logs/teacher/config.yaml"
header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Cookies": "__client=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNsaWVudF8yZVNoZTZ1RnZveG5LWnpNZU4zaWN2cHBReTciLCJyb3RhdGluZ190b2tlbiI6ImFycXF3eGQzbDc4eTN2NWpucjltcHR2N3M1aXZ4bmlrczB2NTNpcmQifQ.HOlF3_taNFGGtBNgJYeTlGlxQhbqSAkUB1SA9YQIo-8Qt2XUpMiPKzYI98fWpxznCfMaWu8-DpQkI7gcheMQNuV_51JwaWFjmglcvVraoeSrGFTFkSnLlvCrAtL1_tGXPZN9bVAi-8mGWNcEecrlLdKGc8ohB1M9Oo6gfyze0AiG4MFGtIO038RMNPEZAses8nNnwIVcUeDuDf2jqYsHFwNvoH6MI_GajC2YqHYkEPbuCDHrd0krJ53NyaG4q6vgwP4tJufhYeYB_x_DTtazkeeHFtpZOMMg_sy2QV0BA1J50pts-Y35gMQfAo5Axx_9Os8n5ZFTWaehleKBeAn0WQ; __client_uat=1711904306; __cf_bm=nhzhgYTa3z5SRHWXT9gEw118V3DxxH7UC3yRgjYRT2s-1712219458-1.0.1.1-C0osoMjhJQE2uBS54OVm.nWI1lMJFVvKC_LR6irXjBCBdRJKp7FU3QoKcPPNfGbQ1hKoEl._CBjsC0f.HI.TGg; _cfuvid=OmJiiA8wUE26AKNsZyHiLouJqx2s3pVNxQhSYfjbUuM-1712219458351-0.0.1.1-604800000; mp_26ced217328f4737497bd6ba6641ca1c_mixpanel=%7B%22distinct_id%22%3A%20%22%24device%3A18e956cf3d5ac5-0f2d22ece30e3e-26001a51-144000-18e956cf3d6ac6%22%2C%22%24device_id%22%3A%20%2218e956cf3d5ac5-0f2d22ece30e3e-26001a51-144000-18e956cf3d6ac6%22%2C%22%24initial_referrer%22%3A%20%22%24direct%22%2C%22%24initial_referring_domain%22%3A%20%22%24direct%22%7D",
    "Origin": "https://suno.com",
    "Referer": "https://suno.com",
    'Content-Type': "text/plain;charset=UTF-8"
}
token=""


#read GPU info
ngpu=torch.cuda.device_count()
gpu_infos=[]
if(torch.cuda.is_available() is False or ngpu==0):
    if_gpu_ok=False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name=torch.cuda.get_device_name(i)
        if("MX"in gpu_name):
            continue
        if("RTX" in gpu_name.upper() or "10"in gpu_name or "16"in gpu_name or "20"in gpu_name or "30"in gpu_name or "40"in gpu_name or "A50"in gpu_name.upper() or "70"in gpu_name or "80"in gpu_name or "90"in gpu_name or "M4"in gpu_name or"P4"in gpu_name or "T4"in gpu_name or "TITAN"in gpu_name.upper()):#A10#A100#V100#A40#P40#M40#K80
            if_gpu_ok=True
            gpu_infos.append("%s\t%s"%(i,gpu_name))
gpu_info="\n".join(gpu_infos)if if_gpu_ok is True and len(gpu_infos)>0 else "很遗憾您这没有能用的显卡来支持您训练"
gpus="-".join([i[0]for i in gpu_infos])


#read cuda info for inference
cuda = {}
min_vram = 0
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        current_vram = torch.cuda.get_device_properties(i).total_memory
        min_vram = current_vram if current_vram > min_vram else min_vram
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"
total_vram = round(min_vram * 9.31322575e-10) if min_vram != 0 else 0
auto_batch = total_vram - 2 if total_vram <= 12 and total_vram > 0 else total_vram
print(f"Current vram: {total_vram} GiB, recommended batch size: {auto_batch}")

#Check BF16 support
amp_options = ["fp32", "fp16"]
if if_gpu_ok:
    if torch.cuda.is_bf16_supported():
        amp_options = ["fp32", "fp16", "bf16"]


def load_model_func(model_path,config_path,total_steps,teacher_or_not,using_device):
    global model,spk_list
    device = cuda[using_device] if "CUDA" in using_device else using_device
    svc_model = Svc(model_path,
                config_path,
                total_steps,
                teacher_or_not,
                device)
    spk_list=svc_model.diffusion_args.spk
    model=svc_model
    return "finish"

def inference_fn(svc_model,spk_list,input_audio,pitch,teacher_or_not,step):
    clip = 0
    slice_db =-40
    pad_seconds = 0.5
    wav_format="wav"
    if teacher_or_not == "teacher":
        isdiffusion = "teacher"
    else:
        isdiffusion = "como"
    resultfolder=f'result_{isdiffusion}'
    if not os.path.exists(resultfolder):
        os.makedirs(resultfolder)
    for spk in spk_list:
        kwarg = {
            "raw_audio_path": input_audio,
            "spk": spk,
            "tran": pitch,
            "slice_db": slice_db,  # -40
            "pad_seconds": pad_seconds,  # 0.5
            "clip_seconds": clip,  # 0
        }
        audio=svc_model.slice_inference(**kwarg)
        song_name=os.path.splitext(os.path.basename(input_audio))[0]
        output_file_name = f'{song_name}_{spk}_{isdiffusion}_{step}.{wav_format}'
        output_file_path = os.path.join("./comosvc/",resultfolder,output_file_name)
        soundfile.write(output_file_path, audio, svc_model.target_sample, format=wav_format)
        svc_model.clear_empty()
    return output_file_path


def vc_fn(input_audio,pitch,teacher_or_not,model_name):
        global model,spk_list
        if input_audio is None:
            return "你还没有上传音频", None
        if model is None:
            return "你还没有加载模型", None
        step = os.path.splitext(os.path.basename(model_name))[0]
        output_file_path = inference_fn(model,spk_list,input_audio,pitch,teacher_or_not,step)
        return output_file_path,"Success"


def vc_batch_fn(input_audio_files,pitch,teacher_or_not,model_name,progress=gr.Progress()):
    global model,spk_list
    if input_audio_files is None or len(input_audio_files) == 0:
            return "你还没有上传音频"
    if model is None:
            return "你还没有加载模型"
    step=os.path.splitext(os.path.basename(model_name))[0]
    _output = []
    for file_obj in progress.tqdm(input_audio_files, desc="Inferencing"):
        print(f"Start processing: {file_obj.name}")
        input_audio_path = file_obj.name
        output_file_path = inference_fn(model,spk_list,input_audio_path,pitch,teacher_or_not,step)
        _output.append(output_file_path)
    return f"批量推理完成，音频已经被保存到result_{teacher_or_not}文件夹"

def tts_fn(_text, _gender, _lang, _rate, _volume,teacher_or_not):
    global model
    if model is None:
         return "你还没有加载模型", None
    pitch=0
    step=0
    _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
    _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
    if _lang == "Auto":
        _gender = "Male" if _gender == "男" else "Female"
        subprocess.run(["python", "tts.py", _text, _lang, _rate, _volume, _gender])
    else:
        subprocess.run(["python", "tts.py", _text, _lang, _rate, _volume])
    input_audio = "tts.wav"
    output_file_path = inference_fn(model,spk_list,input_audio,pitch,teacher_or_not,step)
    os.remove("tts.wav")
    return output_file_path,"Success"

def get_token():
    global token
    load_dotenv()
    suno_auth = suno_api.SunoCookie()
    suno_auth.set_session_id(os.getenv("SESSION_ID"))
    suno_auth.load_cookie(os.getenv("Cookie"))
    suno_api.keep_alive(suno_auth)
    token= suno_api.token
    return "token had refreshed!"

async def fetch(url,header,data=None,method="POST"):
    if data is not None:
        data = json.dumps(data)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.request(method=method, url=url, data=data,headers=header) as resp:
                return await resp.json()
        except Exception as e:
            return f"An error occurred: {e}"



async def suno_generate_CustomMode(title,prompt,tags):
    global header,token
    url="https://studio-api.suno.ai/api/generate/v2/"
    data={
            "prompt":None,
            "tags": None,
            "mv": "chirp-v3-0",
            "title": None,
            "continue_clip_id": None,
            "continue_at": None,
        }
    data.update({"prompt":prompt,"title":title,"tags":tags})
    Authorization=f"Bearer {token}"
    header.update({"Authorization":Authorization})
    text= await fetch(url,header,data)
    return text


async def suno_generate_DescriptionMode(prompt,make_instrumental):
    global header, token
    url="https://studio-api.suno.ai/api/generate/v2/"
    data={
        "prompt":None,
        "make_instrumental":False,
        "mv":"chirp-v3-0"
    }
    data.update({"prompt":prompt,"make_instrumental":make_instrumental})
    Authorization=f"Bearer {token}"
    header.update({"Authorization":Authorization})
    text=await fetch(url,header,data)
    return text



async def get_songs_ids():
    global header, token
    Authorization = f"Bearer {token}"
    header.update({"Authorization": Authorization})
    api_url = "https://studio-api.suno.ai/api/feed"
    songs=await fetch(api_url, header, method="GET")
    # 提取信息
    songs_list=''
    for song in songs:
        if song['title']:
            song_name = song['title']
        else:
            song_name = song['metadata']['prompt']
        song_id = song['id']
        output_song_ids = f"歌名: {song_name}   ID:{song_id}\n"
        songs_list+=output_song_ids
    return songs_list


async def show_music(ids):
    global header,token
    Authorization=f"Bearer {token}"
    header.update({"Authorization":Authorization})
    api_url = f"https://studio-api.suno.ai/api/feed/?ids={ids}"
    response = await fetch(api_url, header, method="GET")
    data=response[0]
    audio_url=data['audio_url']
    return audio_url

    # 定义一个函数来扫描.pt和.yaml文件
def scan_local_models(model_kind):
    pt_files = []
    yaml_files = []
    root_path= os.path.join(local_model_root,model_kind)
    # 在root_path及其所有子目录中搜索所有.pt和.yaml文件
    for pt_file in glob.glob(os.path.join(root_path, '**', '*.pt'), recursive=True):
        pt_files.append(pt_file)
    for yaml_file in glob.glob(os.path.join(root_path, '**', '*.yaml'), recursive=True):
        yaml_files.append(yaml_file)
    # 返回文件路径列表
    return pt_files, yaml_files


def local_model_refresh_fn(model_kind):
    # 调用 scan_local_models 函数并获取两个返回值
    pt_files, yaml_files = scan_local_models(model_kind)
    # 返回两个下拉框的更新
    return gr.Dropdown(choices=pt_files,interactive=True), gr.Dropdown(choices=yaml_files,interactive=True)

def slice_audio(file_format, slice_length):
    files = glob('.comosvc/dataset_slice/*/*.' + file_format)
    for file in files:
        process(file,file_format,slice_length)
    return "音频已切片"

def preprocess():
    subprocess.run(["python","./comosvc/preprocessing1_resample.py"],check=True,capture_output=True, text=True)
    progress = "预处理1已完成。\n"
    subprocess.run(["python", "./comosvc/preprocessing2_flist.py"], check=True,capture_output=True, text=True)
    progress += "预处理2已完成。\n"
    subprocess.run(["python", "./comosvc/preprocessing3_feature.py"], check=True,capture_output=True, text=True)
    progress += "预处理3已完成。\n"
    return progress



def train(model,config_path,model_path):

    # 根据模型选择设置命令行参数
    model_select = '-t' if model == 'consistency model' else ''
     # 设置配置文件和模型文件的路径参数
    config_select = f'-c {config_path}' if config_path else ''
    model_Path= f'-p {model_path}' if model_path else ''

    # 构建激活conda环境并运行训练命令
    activate_command = f"python ./comosvc/train.py {model_select} {config_select} {model_Path} "
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", activate_command], shell=True, env=os.environ.copy())
    return "训练中"


def find_audio_files():
    audio_files = []
    # 支持的音频文件扩展名
    audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
    # 遍历目录
    for root, dirs, files in os.walk('.comosvc/raw'):
        for file in files:
            # 检查文件是否有支持的音频扩展名
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                # 将文件添加到列表中
                audio_files.append(os.path.join(file))
    # 返回音频文件列表
    return audio_files

# 使用 Blocks API 创建应用
app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("训练") as train_tab:
            gr.Markdown(value="""
                请自行准备歌手的清唱录音数据，随后按照如下操作。
                带切片的数据准备流程:
                请将你的原始数据集放在 `dataset_slice` 目录下,然后进行切片操作（注意：切片时间较长，不推荐）
                不带切片的数据准备流程：
                你可以只将数据集放在 `dataset_raw` 目录下，按照以下文件结构:                      
                ### 
                    dataset_raw
                    ├─── speaker0
                    │     ├─── xxx1-xxx1.wav
                    │     ├─── ...
                    │     └─── Lxx-0xx8.wav
                    └─── speaker1
                          ├─── xx2-0xxx2.wav
                          ├─── ...
                          └─── xxx7-xxx007.wav
            """)

            with gr.Column():
                input1 = [
                # gr.File(label="上传WAV文件"),  # 文件上传
                gr.Dropdown(choices=['mp3', 'wav', 'flac', 'aac', 'ogg'],value="wav",label="文件输出格式"),  # 下拉框选择音频格式
                gr.Slider(minimum=5000, maximum=20000, value=10000, label="音频切片长度(ms)")  # 滑动条
                ]
                output1 = gr.Textbox(label="运行结果")
                btn1 = gr.Button("音频剪切")
                btn1.click(slice_audio, inputs=input1, outputs=output1)

            with gr.Column():
                gr.Markdown("音频预处理")
                run_button = gr.Button("运行")
                output_text = gr.Textbox(label="预处理输出")
                run_button.click(
                    fn=preprocess,
                    outputs=output_text
                )

            with gr.Column():
                gr.Markdown("训练")
                btn_2=gr.Button("开始训练")
                input_2=[
                    gr.Dropdown(choices=['teacher model','consistency model'],value="教师模型",label="模型选择"),
                    gr.File(label="选择config文件"),
                    gr.File(label="选择模型文件")
                ]
                imformation=gr.Textbox(label="训练结果输出")
                btn_2.click(
                    fn=train,
                    inputs=input_2,
                    outputs=imformation
                )

        with gr.TabItem("推理") as inference_tab:
            with gr.Row():
                using_device = gr.Dropdown(label="推理设备，默认为自动选择", choices=["Auto", *cuda.keys(), "cpu"],value="Auto")
            with gr.Column():
                gr.Markdown(f'模型应当放置于{local_model_root}文件夹下')
                with gr.Row():
                    teacher_or_not=gr.Dropdown(label='选择训练模型的类型',choices=['teacher','como'],value='teacher')
                with gr.Row():
                    config_path = gr.Dropdown(label='选择config文件夹', choices=[], interactive=True)
                with gr.Row():
                     model_path =gr.Dropdown(label='选择模型文件夹', choices=[], interactive=True)
                local_model_refresh_btn = gr.Button('刷新本地模型列表')
                loadmodel = gr.Button("加载模型", variant="primary")
                unloadmodel = gr.Button("卸载模型", variant="primary")
                total_steps=gr.Slider(minimum=1, maximum=20, value=1, label="训练迭代次数（默认为1）")
            local_model_refresh_btn.click(local_model_refresh_fn, inputs=teacher_or_not, outputs=[model_path, config_path])
            loadmodel.click(load_model_func,[model_path,config_path,total_steps,teacher_or_not,using_device],gr.Textbox())
            unloadmodel.click()
            gr.Markdown(value="""请稍等片刻，模型加载大约需要10秒。后续操作不需要重新加载模型""")
            with gr.Tabs():
                with gr.TabItem("单个音频上传"):
                    vc_input3 = gr.Audio(label="单个音频上传", type="filepath", sources=["upload"])
                with gr.TabItem("批量音频上传"):
                    vc_batch_files = gr.Files(label="批量音频上传", file_types=["audio"], file_count="multiple")
                with gr.TabItem("文字转语音"):
                    gr.Markdown("""文字转语音（TTS）说明：使用edge_tts生成音频，并转换为模型音色。""")
                    text_input = gr.Textbox(label = "在此输入需要转译的文字",)
                    with gr.Row():
                        tts_gender = gr.Radio(label = "说话人性别", choices = ["男","女"], value = "男")
                        tts_lang = gr.Dropdown(label = "选择语言，Auto为根据输入文字自动识别", choices=SUPPORTED_LANGUAGES, value = "Auto")
                    with gr.Row():
                        tts_rate = gr.Slider(label = "TTS语音变速（倍速相对值）", minimum = -1, maximum = 3, value = 0, step = 0.1)
                        tts_volume = gr.Slider(label = "TTS语音音量（相对值）", minimum = -1, maximum = 1.5, value = 0, step = 0.1)
            with gr.Row():
                vc_btn=gr.Button("单个音频推理")
                vc_batch_files_btn =gr.Button("批量音频推理")
                tts_btn=gr.Button("文字转语音")
            pitch = gr.Slider(label="音调调整（+1表示升高一个半音）", minimum=-14, maximum=14, value=0)
            vc_batch_files_btn.click(vc_batch_fn,[vc_batch_files,pitch,teacher_or_not,model_path],gr.Textbox())
            vc_btn.click(vc_fn,[vc_input3,pitch,teacher_or_not,model_path],[gr.Audio(),gr.Textbox()])
            tts_btn.click(tts_fn,[text_input,tts_gender,tts_lang,tts_rate,tts_volume,teacher_or_not],[gr.Audio(),gr.Textbox()])


        with gr.TabItem("其他小部件"):
              with gr.Tab("suno_api"):
                gr.Markdown("suno的api实现")
                with gr.Column():
                    token_refresh_btn=gr.Button("refresh_token")
                    output_suno=gr.Textbox(label="message")
                    token_refresh_btn.click(get_token,outputs=output_suno)
                with gr.Column():
                    with gr.Column():
                        gr.Markdown(value="generate_CustomMode")
                        prompt_custommode=gr.Textbox(label="prompt")
                        title_custommode=gr.Textbox(label="title")
                        tags_custommode=gr.Textbox(label="tags")
                        suno_generate_CustomMode_btn=gr.Button("CustomMode")
                        suno_generate_CustomMode_btn.click(suno_generate_CustomMode,[title_custommode,prompt_custommode,tags_custommode],gr.Textbox())
                    with gr.Column():
                        gr.Markdown(value="generate_DescriptionMode")
                        prompt_descriptionmode = gr.Textbox(label="prompt")
                        make_instrumental=gr.Dropdown(choices=[False,True],value=False,label="only_instrument")
                        suno_generate_DescriptionMode_btn = gr.Button("DescriptionMode")
                        suno_generate_DescriptionMode_btn.click(suno_generate_DescriptionMode,[prompt_descriptionmode,make_instrumental], gr.Textbox())
                with gr.Column():
                    with gr.Column():
                        gr.Markdown(value="get_music_ids")
                        output_ids=gr.Textbox()
                        get_songs_ids_btn=gr.Button("get music ids")
                        get_songs_ids_btn.click(get_songs_ids,outputs=output_ids)
                    with gr.Column():
                        gr.Markdown(value="show music")
                        input_ids=gr.Textbox(value="song_id")
                        output_audio=gr.Audio()
                        show_music_btn = gr.Button("show music")
                        show_music_btn.click(show_music,inputs=input_ids,outputs=output_audio)

    # 启动应用
    app.launch(inbrowser=True)