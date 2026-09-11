// react-mic ships no TypeScript definitions. Minimal typings for the props we use.
declare module "react-mic" {
  import type { Component } from "react";

  export interface ReactMicStopEvent {
    blob: Blob;
    blobURL: string;
    startTime: number;
    stopTime: number;
    option: Record<string, unknown>;
  }

  export interface ReactMicProps {
    record: boolean;
    className?: string;
    onStop?: (recordedBlob: ReactMicStopEvent) => void;
    onData?: (recordedBlob: Blob) => void;
    onStart?: () => void;
    strokeColor?: string;
    backgroundColor?: string;
    mimeType?: string;
    visualSetting?: "sinewave" | "frequencyBars";
    echoCancellation?: boolean;
    autoGainControl?: boolean;
    noiseSuppression?: boolean;
    channelCount?: number;
    bitRate?: number;
    sampleRate?: number;
    timeSlice?: number;
    width?: number;
    height?: number;
  }

  export class ReactMic extends Component<ReactMicProps> {}
}
