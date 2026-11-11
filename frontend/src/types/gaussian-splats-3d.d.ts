declare module "@mkkellogg/gaussian-splats-3d" {
  import type * as THREE from "three";

  export enum SceneFormat {
    Splat = 0,
    KSplat = 1,
    Ply = 2,
    Spz = 3,
  }

  export enum LogLevel {
    None = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
  }

  export enum SceneRevealMode {
    Default = 0,
    Gradual = 1,
    Instant = 2,
  }

  export interface ViewerOptions {
    rootElement?: HTMLElement;
    cameraUp?: [number, number, number];
    initialCameraPosition?: [number, number, number];
    initialCameraLookAt?: [number, number, number];
    sharedMemoryForWorkers?: boolean;
    dynamicScene?: boolean;
    selfDrivenMode?: boolean;
    useBuiltInControls?: boolean;
    logLevel?: LogLevel;
    sceneRevealMode?: SceneRevealMode;
  }

  export interface SplatSceneOptions {
    path?: string;
    format?: SceneFormat;
    splatAlphaRemovalThreshold?: number;
    showLoadingUI?: boolean;
    position?: [number, number, number];
    rotation?: [number, number, number, number];
    scale?: [number, number, number];
  }

  export class Viewer {
    camera: THREE.PerspectiveCamera;
    controls: {
      target: THREE.Vector3;
      update: () => void;
    };
    splatMesh?: {
      getSplatCount(): number;
      getSplatCenter(globalSplatIndex: number, outCenter: THREE.Vector3): void;
    };
    constructor(options?: ViewerOptions);
    addSplatScene(path: string, options?: SplatSceneOptions): Promise<void>;
    start(): void;
    dispose(): Promise<void>;
  }
}

