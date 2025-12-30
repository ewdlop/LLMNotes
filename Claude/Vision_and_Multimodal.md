# Vision and Multimodal Capabilities in Claude

## Overview

Starting with the Claude 3 family (March 2024), Claude gained native vision capabilities, making it a truly multimodal AI system. This enables Claude to process and understand images alongside text, opening up new applications in visual analysis, document processing, UI/UX design, and more.

## Supported Vision Models

### Claude 3 Family (All with Vision)
- **Claude 3 Opus**: Most capable vision understanding
- **Claude 3 Sonnet**: Balanced vision performance and speed
- **Claude 3 Haiku**: Fast vision processing at lower cost

### Claude 3.5 Family (Enhanced Vision)
- **Claude 3.5 Sonnet**: Improved visual reasoning and understanding
- **Claude 3.5 Haiku**: Efficient vision processing with strong performance

### Claude 4 Family
- **Claude Opus 4.5**: State-of-the-art multimodal understanding
- **Claude Sonnet 4.5**: Advanced vision for coding and technical tasks

## Supported Image Formats

### Input Formats
- **PNG** (.png)
- **JPEG** (.jpg, .jpeg)
- **WebP** (.webp)
- **GIF** (.gif) - non-animated

### Input Methods
1. **Base64 Encoding**: Embed images directly in API requests
2. **Image URLs**: Reference publicly accessible images (not all SDKs)
3. **Local Files**: Read and encode local image files

### Size Limitations
- **Maximum file size**: ~25 MB per image (varies by encoding)
- **Maximum dimensions**: Recommended under 8000 pixels per side
- **Optimal resolution**: 1568 pixels (Claude automatically resizes)
- **Token cost**: Varies by image size and detail

## Image Processing Capabilities

### 1. Visual Understanding

**Object Recognition**
- Identify objects, people, animals in images
- Count items and describe spatial relationships
- Recognize brands, logos, and symbols
- Detect text in images (OCR capabilities)

**Scene Understanding**
- Describe overall scene composition
- Identify setting and context
- Recognize activities and events
- Understand mood and atmosphere

**Example**:
```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe everything you see in this image in detail."
                }
            ],
        }
    ],
)
```

### 2. Document Processing

**PDF Analysis**
- Extract text from PDF pages
- Understand document structure
- Analyze forms and tables
- Process receipts and invoices

**Handwriting Recognition**
- Read handwritten notes
- Interpret sketches and diagrams
- Process forms filled by hand
- Transcribe historical documents

**Text Extraction (OCR)**
- Extract text from images
- Maintain formatting and structure
- Handle multiple languages
- Process low-quality scans

**Example**:
```python
# Process a receipt image
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": receipt_image,
                    },
                },
                {
                    "type": "text",
                    "text": """Extract the following information from this receipt:
                    1. Store name
                    2. Date and time
                    3. Items purchased with prices
                    4. Subtotal, tax, and total
                    5. Payment method

                    Format as JSON."""
                }
            ],
        }
    ],
)
```

### 3. Data Visualization Analysis

**Charts and Graphs**
- Interpret bar charts, line graphs, pie charts
- Extract data points and trends
- Compare multiple visualizations
- Identify patterns and anomalies

**Tables and Spreadsheets**
- Parse tabular data from images
- Understand row/column relationships
- Extract specific data points
- Convert to structured formats

**Example**:
```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": chart_image,
                    },
                },
                {
                    "type": "text",
                    "text": "What are the key trends in this chart? What conclusions can we draw?"
                }
            ],
        }
    ],
)
```

### 4. UI/UX Analysis

**Interface Design**
- Analyze website screenshots
- Review mobile app interfaces
- Identify usability issues
- Suggest improvements

**Wireframes and Mockups**
- Interpret design mockups
- Generate code from designs
- Provide design feedback
- Compare design variants

**Accessibility Review**
- Check color contrast
- Identify accessibility issues
- Suggest WCAG compliance improvements
- Review responsive layouts

**Example**:
```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": ui_screenshot,
                    },
                },
                {
                    "type": "text",
                    "text": """Analyze this UI design and provide:
                    1. Accessibility issues
                    2. Usability concerns
                    3. Design best practices violations
                    4. Specific recommendations for improvement"""
                }
            ],
        }
    ],
)
```

### 5. Code and Technical Diagrams

**Architecture Diagrams**
- Interpret system architecture diagrams
- Understand component relationships
- Explain technical workflows
- Identify design patterns

**Code Screenshots**
- Read code from images
- Debug code in screenshots
- Explain code functionality
- Suggest improvements

**Technical Drawings**
- Analyze circuit diagrams
- Interpret engineering drawings
- Understand flowcharts
- Process UML diagrams

**Example**:
```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": architecture_diagram,
                    },
                },
                {
                    "type": "text",
                    "text": "Explain this system architecture. What are potential bottlenecks?"
                }
            ],
        }
    ],
)
```

## Advanced Multimodal Use Cases

### 1. Multiple Image Analysis

Process multiple images in a single request:

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Compare these two product designs:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": design_a,
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": design_b,
                    },
                },
                {
                    "type": "text",
                    "text": "Which design is better and why?"
                }
            ],
        }
    ],
)
```

### 2. Image-to-Code Generation

Convert designs directly to code:

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    system="You are an expert frontend developer.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": ui_mockup,
                    },
                },
                {
                    "type": "text",
                    "text": """Generate React component code for this design.
                    Include:
                    - TypeScript
                    - Tailwind CSS
                    - Responsive design
                    - Accessibility attributes"""
                }
            ],
        }
    ],
)
```

### 3. Visual QA Across Conversation

Maintain context about images:

```python
# First message with image
message1 = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": product_image,
                    },
                },
                {
                    "type": "text",
                    "text": "What product is this?"
                }
            ],
        }
    ],
)

# Follow-up questions reference the same image
message2 = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        # ... previous message context ...
        {
            "role": "user",
            "content": "What are its key features?"
        }
    ],
)
```

### 4. Document Understanding Pipeline

```python
def process_document(image_paths):
    """Process multi-page document"""
    content = [{"type": "text", "text": "Analyze this document:"}]

    for path in image_paths:
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data,
            },
        })

    content.append({
        "type": "text",
        "text": """Provide:
        1. Document summary
        2. Key points and findings
        3. Action items
        4. Structured data extraction"""
    })

    return client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )
```

## Computer Use (Beta)

Claude 3.5 Sonnet introduced experimental "computer use" capabilities:

### What It Can Do
- View screenshots of desktop
- Move mouse cursor
- Click buttons and links
- Type text into applications
- Navigate user interfaces
- Complete multi-step tasks

### Use Cases
- Automated testing
- RPA (Robotic Process Automation)
- User interface navigation
- Data entry automation
- Quality assurance

### Example
```python
# This is a simplified example - actual implementation requires special tools
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    tools=[
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_width_px": 1920,
            "display_height_px": 1080,
        }
    ],
    messages=[
        {
            "role": "user",
            "content": "Open the calculator app and compute 157 * 384"
        }
    ],
)
```

**Important Notes**:
- Still in beta, may have limitations
- Requires special setup and permissions
- Best for controlled environments
- Review security implications

## Vision Performance Benchmarks

### MMMU (Multimodal Understanding)
- **Claude 3 Opus**: 59.4%
- **Claude 3.5 Sonnet**: 68.3%
- Tests college-level subject knowledge with images

### Visual Question Answering
- Strong performance on VQA datasets
- Accurate object counting
- Spatial relationship understanding
- Complex scene comprehension

### OCR Accuracy
- High accuracy on printed text
- Good handwriting recognition
- Multilingual text extraction
- Robust to image quality variations

## Best Practices

### 1. Image Quality
- Use clear, well-lit images
- Avoid extreme compression
- Ensure text is readable
- Optimize resolution (around 1568px)

### 2. Prompt Engineering
- Be specific about what to analyze
- Provide context when needed
- Ask targeted questions
- Request structured output formats

### 3. Token Management
- Images consume tokens based on size
- Resize images appropriately
- Consider cost for multiple images
- Use image caching when available

### 4. Error Handling
```python
try:
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[...]
    )
except anthropic.BadRequestError as e:
    # Handle invalid image format, size issues
    print(f"Bad request: {e}")
except anthropic.APIError as e:
    # Handle API errors
    print(f"API error: {e}")
```

### 5. Privacy Considerations
- Don't send sensitive personal information
- Redact PII from images when possible
- Follow data protection regulations
- Review privacy policies

## Limitations

### Current Limitations
- Cannot generate or edit images (text-to-image)
- May struggle with very small text
- Limited on highly technical diagrams
- Can't play videos or animations
- May misinterpret ambiguous visuals

### Accuracy Considerations
- Verify critical information
- Cross-check extracted data
- Test edge cases thoroughly
- Monitor for errors in production

## Future Developments

Expected improvements in future models:
- Higher resolution processing
- Better video understanding
- Enhanced technical diagram interpretation
- Improved handwriting recognition
- More efficient token usage
- Faster processing speeds

## Resources

### Documentation
- Official Vision Guide: https://docs.anthropic.com/claude/docs/vision
- API Reference: https://docs.anthropic.com/claude/reference
- Best Practices: https://docs.anthropic.com/claude/docs/vision-best-practices

### Examples
- Cookbook: https://github.com/anthropics/anthropic-cookbook
- Sample applications
- Code templates
- Interactive demos

---

**Last Updated**: December 2024
**Status**: Production-ready across Claude 3, 3.5, and 4 families
