import google.generativeai as genai
from google.colab import userdata

import google.colab as colab
api=colab.userdata.get("GEMINI_API_KEY")

import google.genai as gen
from google.genai import types

class ChatGemini():
  def __init__(self, chat_model:str="gemini-1.5-flash",gen_model:str="imagen-3.0-generate-002", api_key:str=api, instructions:str=None,history:bool=False, max_output_tokens:int=200, temperature:float=0.1, code_execution:bool=False):
    '''
    initializes Chat object with Gemini.
    Args:
      chat_model: Model to use for text responses.
      gen_model: Model to use for image generation.
      api_key: API key provided by Google Studio.
      instructions: Special Instructions for the model to follow.
      max_output_tokens: Maximum number of tokens model can output.
      temperature: Percentage of randomness in model's reponse.
      code_execution: Set it to True if you want the model to provide and 
      execute code for you. (max 30 secs)
    '''
    genai.configure(api_key=api)
    self.c_model = genai.GenerativeModel(model_name=chat_model, system_instruction=instructions, generation_config=genai.GenerationConfig(max_output_tokens=max_output_tokens,temperature=temperature), tools = "code_execution" if code_execution else None)
    
    self.g_model = gen.Client(api_key=api_key)
    self.history=history
    self.history_=[]
    self.gen_model = gen_model

  def chat(self, text:str, image_path:str = None):
    '''
    This function lets you chat with Gemini.
    
    Args:
      text: Prompt
      image_path: The image filepath you want to provide as input. 
    Returns:
      response: The model's response.
    '''
    if image_path is not None:
      input=[PIL.Image.open("/content/i9c3o5nq.png"), text]
    else:
      input=text
    if not self.history:
      response=self.c_model.generate_content(input)
      return response.text
    else:
      chat = self.c_model.start_chat(history=self.history_)
      response=chat.send_message(input)
      return response.text
  def generate(self, prompt:str, neg_prompt:str=None, n_images=1, allow_people_image=False, aspect_ratio:str="1:1"):
    '''
    Generates n_images using imagen model using prompt.

    Args:
      prompt: Description of image
      neg_prompt: A description of what you want to omit in the generated images.
      n_images: No. of images to generate.
      allow_people_image: Allow people generation in pictures.
      aspect_ratio: Aspect ratio of generated images.
    Returns:
      Images generated...
    '''
    response = self.g_model.models.generate_image(
        model=self.gen_model,
        prompt=prompt,
        config=gen.types.GenerateImageConfig(
            negative_prompt=neg_prompt,
            number_of_images=n_images
        ))
    return response.generated_images[0].image.show()
