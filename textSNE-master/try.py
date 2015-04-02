# -*- coding: utf-8 -*-
import Image
import ImageFont, ImageDraw
image=Image.new("RGB",[320,320])
draw = ImageDraw.Draw(image)
a=u"कामयाब"
font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-devanagari-fonts/nakula.ttf",14)
draw.text((50, 50), a,font=font)
image.save("a.png")
