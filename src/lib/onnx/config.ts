export interface SamOnnxConfig {
  encoderUrl: string;
  decoderUrl: string;
  inputSize: number;
  encoderInput?: string;
  encoderOutput?: string;
  decoderEmbeddingInput?: string;
  decoderPointCoordsInput?: string;
  decoderPointLabelsInput?: string;
  decoderMaskInput?: string;
  decoderHasMaskInput?: string;
  decoderOriginalSizeInput?: string;
  decoderMaskOutput?: string;
  decoderScoreOutput?: string;
}

function env(name: keyof ImportMetaEnv): string | undefined {
  const value = import.meta.env[name];
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

export const samOnnxConfig: SamOnnxConfig = {
  encoderUrl: env('VITE_ENCODER_MODEL_URL') ?? '/models/mobile_sam_encoder.onnx',
  decoderUrl: env('VITE_DECODER_MODEL_URL') ?? '/models/mobile_sam_decoder.onnx',
  inputSize: Number(env('VITE_MODEL_INPUT_SIZE') ?? 1024),
  encoderInput: env('VITE_ENCODER_INPUT_NAME'),
  encoderOutput: env('VITE_ENCODER_OUTPUT_NAME'),
  decoderEmbeddingInput: env('VITE_DECODER_EMBEDDING_INPUT_NAME'),
  decoderPointCoordsInput: env('VITE_DECODER_POINT_COORDS_INPUT_NAME'),
  decoderPointLabelsInput: env('VITE_DECODER_POINT_LABELS_INPUT_NAME'),
  decoderMaskInput: env('VITE_DECODER_MASK_INPUT_NAME'),
  decoderHasMaskInput: env('VITE_DECODER_HAS_MASK_INPUT_NAME'),
  decoderOriginalSizeInput: env('VITE_DECODER_ORIGINAL_SIZE_INPUT_NAME'),
  decoderMaskOutput: env('VITE_DECODER_MASK_OUTPUT_NAME'),
  decoderScoreOutput: env('VITE_DECODER_SCORE_OUTPUT_NAME'),
};
