/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type AppView = 'Live' | 'Upload' | 'History' | 'Analytics' | 'Settings';

export type RiskLevel = 'SAFE' | 'WARNING' | 'DANGER';
export type VisualMode = 'RGB' | 'GRAYSCALE' | 'NIGHT' | 'THERMAL';

export interface Detection {
  id: string;
  label: string;
  confidence: number;
  timestamp: number;
  type: 'Object' | 'Text' | 'Scene';
  distance?: number; // Estimated distance in meters
  risk?: RiskLevel;
  box?: { x: number, y: number, w: number, h: number };
}

export interface MediaAnalysis {
  id: string;
  type: 'image' | 'video';
  url: string;
  name: string;
  detections: Detection[];
  summary: string;
  timestamp: number;
}

export interface VoiceSettings {
  enabled: boolean;
  voice: string;
  pitch: number;
  rate: number;
}
