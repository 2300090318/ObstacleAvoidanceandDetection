import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

export async function analyzeMedia(file: File, type: 'image' | 'video') {
  const model = "gemini-3-flash-preview";
  
  const base64Data = await new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = (reader.result as string).split(',')[1];
      resolve(base64);
    };
    reader.readAsDataURL(file);
  });

  const prompt = type === 'image' 
    ? "Analyze this image. List all significant objects detected, their confidence (0-1), and a brief summary of the scene. Return in JSON format with 'detections' (array of {label, confidence, type}) and 'summary' (string)."
    : "Analyze this video. List significant events or objects detected, their confidence, and a summary. Return in JSON format.";

  try {
    const response = await ai.models.generateContent({
      model,
      contents: [
        {
          parts: [
            { text: prompt },
            {
              inlineData: {
                data: base64Data,
                mimeType: file.type
              }
            }
          ]
        }
      ],
      config: {
        responseMimeType: "application/json"
      }
    });

    return JSON.parse(response.text || '{}');
  } catch (error) {
    console.error("AI Analysis failed:", error);
    throw error;
  }
}
