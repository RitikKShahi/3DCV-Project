// Web Worker for background dataset warm-start processing
// Runs off the main thread so the browser never freezes and never triggers "Page Unresponsive".

importScripts("/jszip.min.js");

const C0 = 0.28209479177387814;

function plyToPoints3DBin(plyArrayBuffer) {
  const textDecoder = new TextDecoder();
  const bytes = new Uint8Array(plyArrayBuffer);
  
  let headerEnd = -1;
  for (let i = 0; i < Math.min(bytes.length - 11, 4096); i++) {
    if (
      bytes[i] === 101 && bytes[i+1] === 110 && bytes[i+2] === 100 &&
      bytes[i+3] === 95 && bytes[i+4] === 104 && bytes[i+5] === 101 &&
      bytes[i+6] === 97 && bytes[i+7] === 100 && bytes[i+8] === 101 &&
      bytes[i+9] === 114 && bytes[i+10] === 10
    ) {
      headerEnd = i + 11;
      break;
    }
  }

  if (headerEnd === -1) {
    throw new Error("Invalid PLY file: could not find end_header");
  }

  const headerStr = textDecoder.decode(bytes.subarray(0, headerEnd));
  const lines = headerStr.split("\n");

  let vertexCount = 0;
  const properties = [];

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === "element" && parts[1] === "vertex") {
      vertexCount = parseInt(parts[2], 10);
    } else if (parts[0] === "property") {
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  if (vertexCount <= 0) {
    throw new Error("No vertices found in PLY header");
  }

  let stride = 0;
  let xOffset = -1, yOffset = -1, zOffset = -1;
  let fdc0Offset = -1, fdc1Offset = -1, fdc2Offset = -1;
  let rOffset = -1, gOffset = -1, bOffset = -1;

  for (const prop of properties) {
    const size = (prop.type === "double" || prop.type === "float64") ? 8 :
                 (prop.type === "float" || prop.type === "float32" || prop.type === "int") ? 4 :
                 (prop.type === "uchar" || prop.type === "uint8") ? 1 : 4;
    
    if (prop.name === "x") xOffset = stride;
    else if (prop.name === "y") yOffset = stride;
    else if (prop.name === "z") zOffset = stride;
    else if (prop.name === "f_dc_0") fdc0Offset = stride;
    else if (prop.name === "f_dc_1") fdc1Offset = stride;
    else if (prop.name === "f_dc_2") fdc2Offset = stride;
    else if (prop.name === "red") rOffset = stride;
    else if (prop.name === "green") gOffset = stride;
    else if (prop.name === "blue") bOffset = stride;

    stride += size;
  }

  const dataView = new DataView(plyArrayBuffer, headerEnd);
  const outBuffer = new ArrayBuffer(8 + vertexCount * 51);
  const outView = new DataView(outBuffer);

  outView.setBigUint64(0, BigInt(vertexCount), true);

  let outOffset = 8;
  for (let i = 0; i < vertexCount; i++) {
    const vOffset = i * stride;
    const x = (xOffset >= 0) ? dataView.getFloat32(vOffset + xOffset, true) : 0;
    const y = (yOffset >= 0) ? dataView.getFloat32(vOffset + yOffset, true) : 0;
    const z = (zOffset >= 0) ? dataView.getFloat32(vOffset + zOffset, true) : 0;

    let r = 128, g = 128, b = 128;
    if (fdc0Offset >= 0 && fdc1Offset >= 0 && fdc2Offset >= 0) {
      const f0 = dataView.getFloat32(vOffset + fdc0Offset, true);
      const f1 = dataView.getFloat32(vOffset + fdc1Offset, true);
      const f2 = dataView.getFloat32(vOffset + fdc2Offset, true);
      r = Math.min(255, Math.max(0, Math.round((0.5 + C0 * f0) * 255)));
      g = Math.min(255, Math.max(0, Math.round((0.5 + C0 * f1) * 255)));
      b = Math.min(255, Math.max(0, Math.round((0.5 + C0 * f2) * 255)));
    } else if (rOffset >= 0 && gOffset >= 0 && bOffset >= 0) {
      r = dataView.getUint8(vOffset + rOffset);
      g = dataView.getUint8(vOffset + gOffset);
      b = dataView.getUint8(vOffset + bOffset);
    }

    outView.setBigUint64(outOffset, BigInt(i + 1), true); outOffset += 8;
    outView.setFloat64(outOffset, x, true); outOffset += 8;
    outView.setFloat64(outOffset, y, true); outOffset += 8;
    outView.setFloat64(outOffset, z, true); outOffset += 8;
    outView.setUint8(outOffset, r); outOffset += 1;
    outView.setUint8(outOffset, g); outOffset += 1;
    outView.setUint8(outOffset, b); outOffset += 1;
    outView.setFloat64(outOffset, 0.0, true); outOffset += 8;
    outView.setBigUint64(outOffset, 0n, true); outOffset += 8;
  }

  return outBuffer;
}

self.onmessage = async function(e) {
  const { datasetZip, plyBuffer } = e.data;

  try {
    self.postMessage({ type: "PROGRESS", text: "Converting 3D splats to point cloud..." });
    const points3DBin = plyToPoints3DBin(plyBuffer);

    self.postMessage({ type: "PROGRESS", text: "Unpacking dataset manifest..." });
    const zip = await JSZip.loadAsync(datasetZip);

    let targetPath = null;
    zip.forEach((relativePath) => {
      if (relativePath.endsWith("sparse/0/points3D.bin") || relativePath.endsWith("sparse/0/points3D.txt")) {
        targetPath = relativePath;
      }
    });

    if (!targetPath) {
      zip.forEach((relativePath) => {
        if (relativePath.includes("sparse/0/")) {
          const root = relativePath.split("sparse/0/")[0];
          targetPath = root + "sparse/0/points3D.bin";
        }
      });
    }

    if (!targetPath) {
      targetPath = "sparse/0/points3D.bin";
    }

    self.postMessage({ type: "PROGRESS", text: "Injecting trained 3D geometry..." });
    zip.file(targetPath, points3DBin);

    self.postMessage({ type: "PROGRESS", text: "Finalizing warm-start package..." });
    // Using compression: STORE avoids re-compressing 150MB of JPEG images, making it instant!
    const warmBlob = await zip.generateAsync({
      type: "blob",
      compression: "STORE"
    });

    self.postMessage({ type: "DONE", blob: warmBlob });
  } catch (err) {
    self.postMessage({ type: "ERROR", error: err.message || String(err) });
  }
};

