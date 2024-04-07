import google.generativeai as genai
import textwrap
def print_out(text):
    text = text.replace('•', '  *')
    return print(textwrap.indent(text, '> ', predicate=lambda _: True))

generation_config = {
    "temperature":0,
    "max_output_tokens":400,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

GOOGLE_API_KEY="AIzaSyC8_wpSt82IAwDok3AoJiVFNk0Sv7-pfnE"

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(model_name='gemini-pro',
                              generation_config=generation_config,
                              safety_settings=safety_settings)


prompt = '''

### Instruction ###
The given sentence represents image path and source text for a composed image retrieval system. 
For instance, given the image file paths 'dress/B00CL0VWTA.jpg' (source image) and 'dress/B00CF2X97M.jpg' (target image) alongside descriptors like 'is blue with tighter fitting' and 'has sleeves and polka dots,' 
the consolidated description becomes 'is blue with tighter fitting and has sleeves and polka dots.

When provided with such descriptive text, I want you to convert it into a more 'ambiguous form'. 
This means reducing the clarity of one or two attributes due to hypothetical customer uncertainty by changing it with ambiguous expression or by omitting it. 

### Rules ###
And the following are the rules to follow.

0. If initially, the sentence is ambiguous enough, then do not change it.
1. If initially there is only one attribute, such as color (for example, the text is only "is green"), do not alter it; leave it unchanged.
2. To make attributes ambiguous, use terms like 'some', 'other', 'different', 'maybe', 'some length of', 'some color of', kinds of words if needed. 
3. For the case only if the original color is 'black' or 'white' (which is the darkest/lightest color), it can be changed to 'a darker color' or 'a lighter color' respectively.
4. Avoid excessive ambiguity; for instance, if attributes include color and pattern (like "blue" and "dots"), and if the color is made vague ("some color"), ensure the pattern ("dots") remains distinct.
5. If one attribute becomes vague, ensure this change is applied consistently across same image file path.

### Examples ###
Make sure to give the answer with the same structure with the question - 2 sentences.

Q.
shirt/B0083I6W08.jpg;shirt/B00BPD4N5E.jpg;is green with a four leaf clover 
shirt/B0083I6W08.jpg;shirt/B00BPD4N5E.jpg;is green and has no text 
A.
shirt/B0083I6W08.jpg;shirt/B00BPD4N5E.jpg;is green with some patterns 
shirt/B0083I6W08.jpg;shirt/B00BPD4N5E.jpg;is green and has no text 

Q.
shirt/B007VYS9I8.jpg;shirt/B000N9FHLU.jpg;Is green 
shirt/B007VYS9I8.jpg;shirt/B000N9FHLU.jpg;is green
A.
shirt/B007VYS9I8.jpg;shirt/B000N9FHLU.jpg;Is green 
shirt/B007VYS9I8.jpg;shirt/B000N9FHLU.jpg;is green 

Q.
dress/B00FQANLX2.jpg;dress/B007U6KROG.jpg;has thin straps and different pattern 
dress/B00FQANLX2.jpg;dress/B007U6KROG.jpg;more autumn colored and longer
A.
dress/B00FQANLX2.jpg;dress/B007U6KROG.jpg;has different pattern 
dress/B00FQANLX2.jpg;dress/B007U6KROG.jpg;more autumn colored

Q.
toptee/B007S3TENQ.jpg;toptee/B0081U1TNI.jpg;is a short sleeve v neck 
toptee/B007S3TENQ.jpg;toptee/B0081U1TNI.jpg;is short sleeved and pink
A.
toptee/B007S3TENQ.jpg;toptee/B0081U1TNI.jpg;is sleeve v neck 
toptee/B007S3TENQ.jpg;toptee/B0081U1TNI.jpg;is sleeved and pink

Q.
dress/B00CL0VWTA.jpg;dress/B00CF2X97M.jpg;is blue with tighter fitting 
dress/B00CL0VWTA.jpg;dress/B00CF2X97M.jpg;is sleeves and poka dots
A.
dress/B00CL0VWTA.jpg;dress/B00CF2X97M.jpg;is blue
dress/B00CL0VWTA.jpg;dress/B00CF2X97M.jpg;has some patterns

Q.
dress/B0084DJQ6O.jpg;dress/B00BJ1NFNI.jpg;is pink in color 
dress/B0084DJQ6O.jpg;dress/B00BJ1NFNI.jpg;is sleeveless and pink
A.
dress/B0084DJQ6O.jpg;dress/B00BJ1NFNI.jpg;is pink in color 
dress/B0084DJQ6O.jpg;dress/B00BJ1NFNI.jpg;pink

Q.
dress/B0072QV4V4.jpg;dress/B009S3GXWY.jpg;Has lace and is short and red. 
dress/B0072QV4V4.jpg;dress/B009S3GXWY.jpg;is more red and shorter
A.
dress/B0072QV4V4.jpg;dress/B009S3GXWY.jpg;Has lace and is short. 
dress/B0072QV4V4.jpg;dress/B009S3GXWY.jpg;is shorter

Q.
toptee/B008JYNN30.jpg;toptee/B008DVXGO0.jpg;its a striped sweater with a higher nackline 
toptee/B008JYNN30.jpg;toptee/B008DVXGO0.jpg;is sleeveless and has a brighter color pattern 
A.
toptee/B008JYNN30.jpg;toptee/B008DVXGO0.jpg;its a striped sweater
toptee/B008JYNN30.jpg;toptee/B008DVXGO0.jpg;is sleeveless and has a brighter color pattern

Q.
toptee/B001NZVD4I.jpg;toptee/B004L83ZEU.jpg;is lighter in color and has american flag 
toptee/B001NZVD4I.jpg;toptee/B004L83ZEU.jpg;is more patriotic and less designer
A.
toptee/B001NZVD4I.jpg;toptee/B004L83ZEU.jpg;has some flag 
toptee/B001NZVD4I.jpg;toptee/B004L83ZEU.jpg;is more patriotic and less designer

Q.
shirt/B009CCS6JO.jpg;shirt/B00G0IUSB2.jpg;is gray and has a different image 
shirt/B009CCS6JO.jpg;shirt/B00G0IUSB2.jpg;is pale grey with a weather report
A.
shirt/B009CCS6JO.jpg;shirt/B00G0IUSB2.jpg;is gray
shirt/B009CCS6JO.jpg;shirt/B00G0IUSB2.jpg;is pale grey with some weather-related thing
'''

domains = ['dress', 'shirt', 'toptee']

for domain in domains:
    file_path = f"/mnt/sting/jaehyun/dataset/fashionIQ/captions_pairs/fashion_iq-val-cap-{domain}.txt"
    ambiguous_file_path = f"/mnt/sting/jaehyun/dataset/fashionIQ/captions_pairs/ambiguous_fashion_iq-val-cap-{domain}.txt"

    with open(file_path, 'r') as file:
        line_count = len(file.readlines())

    with open(file_path, 'r') as file:
        i = 0
        while i != line_count:
            first_line = file.readline().strip()
            second_line = file.readline().strip()

            source_sentence = f'''
            {first_line}
            {second_line}
            '''

            # source_sentence = '''
            # shirt/B007VYS9I8.jpg;shirt/B000N9FHLU.jpg;Is green
            # shirt/B007VYS9I8.jpg;shirt/B000N9FHLU.jpg;is green
            # '''

            response = model.generate_content(
                f"PROMPT : {prompt}\n Q.\n{source_sentence}\nA.\n")


            with open(ambiguous_file_path, 'a') as target_file:
                target_file.write(response.candidates[0].content.parts[0].text)
                target_file.write('\n')
            print(response.candidates[0].content.parts[0].text)
            assert(len(response.candidates[0].content.parts[0].text.split('\n'))==2)

            i += 2
            # exit(0)

# source_sentence = '''
# dress/B005X4PL1G.jpg;dress/B0084Y8XIU.jpg;is shiny and silver with shorter sleeves
# dress/B005X4PL1G.jpg;dress/B0084Y8XIU.jpg;fit and flare
# dress/B008XODTD0.jpg;dress/B00AKLK08G.jpg;is grey with black design
# dress/B008XODTD0.jpg;dress/B00AKLK08G.jpg;is a light printed short dress
# dress/B00BPYP69K.jpg;dress/B00CMPE0C0.jpg;is a solid red color
# dress/B00BPYP69K.jpg;dress/B00CMPE0C0.jpg;shorter and tighter with more blue and white
# dress/B000QSGNOI.jpg;dress/B004UO3XYC.jpg;is a plain white feminine t shirt
# dress/B000QSGNOI.jpg;dress/B004UO3XYC.jpg;is a tan shirt.
# dress/B00FQANLX2.jpg;dress/B007U6KROG.jpg;has thin straps and different pattern
# dress/B00FQANLX2.jpg;dress/B007U6KROG.jpg;more autumn colored and longer
# dress/B00CL14N3Q.jpg;dress/B004KHOPDW.jpg;is longer and more sculptural
# dress/B00CL14N3Q.jpg;dress/B004KHOPDW.jpg;rushed solid green
# dress/B009CMY4BS.jpg;dress/B0091PLEKA.jpg;is gold and strapless
# dress/B009CMY4BS.jpg;dress/B0091PLEKA.jpg;button front longer sleeves
# '''



# response = model.generate_content(f"{prompt} \n now, the sentence I want to make it ambiguous is : {source_sentence}")
#
# print(response.candidates[0].content.parts[0].text)
# print(file_content)