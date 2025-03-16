from PIL import Image, ImageDraw, ImageFont
import math

def create_gear(draw, center, outer_radius, inner_radius, num_teeth=12):
    # Calculate points for the gear shape
    points = []
    for i in range(num_teeth * 2):
        angle = i * math.pi / num_teeth
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points

# Create a new image with a white background
size = (800, 800)
img = Image.new('RGB', size, 'white')
draw = ImageDraw.Draw(img)

# Colors
navy_blue = (22, 23, 73)
yellow = (230, 229, 8)

# Draw the gear
center = (size[0]//2, size[1]//2)
gear_points = create_gear(draw, center, 350, 300)
draw.polygon(gear_points, fill=navy_blue)

# Draw the sun rays
ray_length = 200
num_rays = 16
for i in range(num_rays):
    angle = i * 2 * math.pi / num_rays
    start_x = center[0] + 150 * math.cos(angle)
    start_y = center[1] + 150 * math.sin(angle)
    end_x = center[0] + ray_length * math.cos(angle)
    end_y = center[1] + ray_length * math.sin(angle)
    draw.line([(start_x, start_y), (end_x, end_y)], fill=yellow, width=20)

# Draw the flame
flame_points = [
    (center[0], center[1] - 100),  # Top
    (center[0] - 60, center[1] + 50),  # Bottom left
    (center[0], center[1]),  # Middle bottom
    (center[0] + 60, center[1] + 50),  # Bottom right
]
draw.polygon(flame_points, fill=yellow)

# Draw the book
book_width = 120
book_height = 80
book_left = center[0] - book_width//2
book_top = center[1] - book_height//2
draw.rectangle([book_left, book_top, book_left + book_width, book_top + book_height], fill='white')
draw.line([center[0], book_top, center[0], book_top + book_height], fill='black', width=2)

# Add text
try:
    # Note: You'll need to provide your own font file or use a system font
    font = ImageFont.truetype("arial.ttf", 40)
    draw.text((center[0], center[1] + 250), "BUTUAN CITY", fill='white', anchor="mm", font=font)
    draw.text((center[0], center[1] + 300), "SCHOOL OF ARTS AND TRADES", fill='white', anchor="mm", font=font)
    draw.text((center[0], center[1] + 350), "1983", fill=yellow, anchor="mm", font=font)
except:
    # Fallback if font not available
    draw.text((center[0], center[1] + 250), "BUTUAN CITY", fill='white', anchor="mm")
    draw.text((center[0], center[1] + 300), "SCHOOL OF ARTS AND TRADES", fill='white', anchor="mm")
    draw.text((center[0], center[1] + 350), "1983", fill=yellow, anchor="mm")

# Save the image
img.save('static/logo.png', 'PNG')
