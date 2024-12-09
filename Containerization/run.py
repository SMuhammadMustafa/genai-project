from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.load_lora_weights("posters.safetensors")

prompt = "`A gang of outlaws in a western"
image = pipe(prompt).images[0]

prompt = "Footballer biography"
image = pipe(prompt).images[0]

prompt = "Movie Set in space with spaceships and stars"
image = pipe(prompt).images[0]

prompt = "transformers"
image = pipe(prompt).images[0]