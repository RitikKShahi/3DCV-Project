import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import JSZip from 'jszip';

const C0 = 0.28209479177387814;

function plyToPoints3DBin(plyBuffer: Buffer): Buffer {
  let headerEnd = plyBuffer.indexOf('end_header\n');
  if (headerEnd === -1) throw new Error('Invalid PLY: missing end_header');
  headerEnd += 11;

  const headerStr = plyBuffer.subarray(0, headerEnd).toString('utf8');
  const lines = headerStr.split('\n');
  let vertexCount = 0;
  const properties: Array<{ type: string; name: string }> = [];

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'element' && parts[1] === 'vertex') {
      vertexCount = parseInt(parts[2], 10);
    } else if (parts[0] === 'property') {
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  let stride = 0;
  let xOffset = -1, yOffset = -1, zOffset = -1;
  let fdc0Offset = -1, fdc1Offset = -1, fdc2Offset = -1;

  for (const prop of properties) {
    const size = (prop.type === 'double' || prop.type === 'float64') ? 8 :
                 (prop.type === 'float' || prop.type === 'float32' || prop.type === 'int') ? 4 :
                 (prop.type === 'uchar' || prop.type === 'uint8') ? 1 : 4;
    if (prop.name === 'x') xOffset = stride;
    else if (prop.name === 'y') yOffset = stride;
    else if (prop.name === 'z') zOffset = stride;
    else if (prop.name === 'f_dc_0') fdc0Offset = stride;
    else if (prop.name === 'f_dc_1') fdc1Offset = stride;
    else if (prop.name === 'f_dc_2') fdc2Offset = stride;
    stride += size;
  }

  const outBuf = Buffer.alloc(8 + vertexCount * 51);
  outBuf.writeBigUInt64LE(BigInt(vertexCount), 0);

  let outOffset = 8;
  for (let i = 0; i < vertexCount; i++) {
    const inOffset = headerEnd + i * stride;
    const x = xOffset >= 0 ? plyBuffer.readFloatLE(inOffset + xOffset) : 0;
    const y = yOffset >= 0 ? plyBuffer.readFloatLE(inOffset + yOffset) : 0;
    const z = zOffset >= 0 ? plyBuffer.readFloatLE(inOffset + zOffset) : 0;
    const f0 = fdc0Offset >= 0 ? plyBuffer.readFloatLE(inOffset + fdc0Offset) : 0;
    const f1 = fdc1Offset >= 0 ? plyBuffer.readFloatLE(inOffset + fdc1Offset) : 0;
    const f2 = fdc2Offset >= 0 ? plyBuffer.readFloatLE(inOffset + fdc2Offset) : 0;

    const r = Math.min(255, Math.max(0, Math.round((0.5 + C0 * f0) * 255)));
    const g = Math.min(255, Math.max(0, Math.round((0.5 + C0 * f1) * 255)));
    const b = Math.min(255, Math.max(0, Math.round((0.5 + C0 * f2) * 255)));

    outBuf.writeBigUInt64LE(BigInt(i + 1), outOffset); outOffset += 8;
    outBuf.writeDoubleLE(x, outOffset); outOffset += 8;
    outBuf.writeDoubleLE(y, outOffset); outOffset += 8;
    outBuf.writeDoubleLE(z, outOffset); outOffset += 8;
    outBuf.writeUInt8(r, outOffset); outOffset += 1;
    outBuf.writeUInt8(g, outOffset); outOffset += 1;
    outBuf.writeUInt8(b, outOffset); outOffset += 1;
    outBuf.writeDoubleLE(0.0, outOffset); outOffset += 8;
    outBuf.writeBigUInt64LE(0n, outOffset); outOffset += 8;
  }

  return outBuf;
}

function incrementalApiPlugin() {
  return {
    name: 'incremental-api',
    configureServer(server: any) {
      server.middlewares.use(async (req: any, res: any, next: any) => {
        // 1. Incremental reconstruction endpoint
        if (req.url?.startsWith('/api/incremental') && req.method === 'POST') {
          try {
            console.log('[IncrementalAPI] Processing warm-start dataset request...');
            const chunks: any[] = [];
            for await (const chunk of req) {
              chunks.push(chunk);
            }
            const bodyBuffer = Buffer.concat(chunks);

            const webReq = new Request(`http://${req.headers.host || 'localhost'}${req.url}`, {
              method: 'POST',
              headers: req.headers,
              body: bodyBuffer,
            });

            const formData = await webReq.formData();
            const zipEntry = formData.get('zip') as any;
            const plyEntry = formData.get('ply') as any;

            if (!zipEntry) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: 'Missing zip dataset' }));
              return;
            }

            const zipArrayBuffer = await zipEntry.arrayBuffer();
            const zipBuffer = Buffer.from(zipArrayBuffer);

            let plyBuffer: Buffer | null = null;
            if (plyEntry) {
              plyBuffer = Buffer.from(await plyEntry.arrayBuffer());
            } else if (fs.existsSync('public/truck_sparse.ply')) {
              plyBuffer = fs.readFileSync('public/truck_sparse.ply');
            }

            if (!plyBuffer) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: 'No base PLY model available' }));
              return;
            }

            console.log('[IncrementalAPI] Converting PLY to COLMAP points3D.bin...');
            const points3DBin = plyToPoints3DBin(plyBuffer);

            console.log('[IncrementalAPI] Injecting points3D.bin into ZIP...');
            const zip = await JSZip.loadAsync(zipBuffer);

            let targetPath = 'sparse/0/points3D.bin';
            zip.forEach((relPath) => {
              if (relPath.endsWith('sparse/0/points3D.bin') || relPath.endsWith('sparse/0/points3D.txt')) {
                targetPath = relPath;
              }
            });

            zip.file(targetPath, points3DBin);

            console.log('[IncrementalAPI] Saving warm dataset...');
            const outZipBuf = await zip.generateAsync({
              type: 'nodebuffer',
              compression: 'STORE',
            });

            const outFilePath = path.join(process.cwd(), 'public', 'warm_dataset.zip');
            fs.writeFileSync(outFilePath, outZipBuf);

            console.log('[IncrementalAPI] Successfully created warm_dataset.zip (' + outZipBuf.length + ' bytes)');
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ success: true, url: `/warm_dataset.zip?t=${Date.now()}` }));
            return;
          } catch (err: any) {
            console.error('[IncrementalAPI] Error:', err);
            res.statusCode = 500;
            res.end(JSON.stringify({ error: err.message || String(err) }));
            return;
          }
        }

        // 2. COLMAP Data Preparation endpoint (colmap-gradio inspired)
        if (req.url?.startsWith('/api/colmap/process') && req.method === 'POST') {
          try {
            console.log('[ColmapAPI] Receiving raw photos for COLMAP data preparation...');
            const chunks: any[] = [];
            for await (const chunk of req) {
              chunks.push(chunk);
            }
            const bodyBuffer = Buffer.concat(chunks);
            const webReq = new Request(`http://${req.headers.host || 'localhost'}${req.url}`, {
              method: 'POST',
              headers: req.headers,
              body: bodyBuffer,
            });

            const formData = await webReq.formData();
            const scaling = (formData.get('scaling') as string) || '1600k';
            const matching = (formData.get('matching') as string) || 'Sequential';
            const cameraModel = (formData.get('camera_model') as string) || 'SIMPLE_RADIAL';
            const fps = (formData.get('fps') as string) || '2.0';

            const videoEntry = formData.get('video') as any;
            const photos: any[] = formData.getAll('photos');

            if (!videoEntry && (!photos || photos.length === 0)) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: 'No video or photos uploaded' }));
              return;
            }

            const runId = Date.now().toString(36) + Math.random().toString(36).substring(2, 6);
            const workspaceDir = path.join('/tmp', `colmap_run_${runId}`);
            const rawImagesDir = path.join(workspaceDir, 'raw_images');
            fs.mkdirSync(rawImagesDir, { recursive: true });

            let videoPath: string | null = null;
            if (videoEntry && typeof videoEntry.arrayBuffer === 'function') {
              const videoExt = path.extname(videoEntry.name || '.mp4') || '.mp4';
              videoPath = path.join(workspaceDir, `input_video${videoExt}`);
              const buffer = Buffer.from(await videoEntry.arrayBuffer());
              fs.writeFileSync(videoPath, buffer);
              console.log(`[ColmapAPI] Stored input video to ${videoPath} (${buffer.length} bytes)...`);
            } else if (photos && photos.length > 0) {
              console.log(`[ColmapAPI] Storing ${photos.length} input photos in ${rawImagesDir}...`);
              for (let i = 0; i < photos.length; i++) {
                const file = photos[i];
                const filename = file.name || `photo_${String(i).padStart(4, '0')}.jpg`;
                const buffer = Buffer.from(await file.arrayBuffer());
                fs.writeFileSync(path.join(rawImagesDir, filename), buffer);
              }
            }

            const downloadsDir = path.join(process.cwd(), 'public', 'downloads');
            fs.mkdirSync(downloadsDir, { recursive: true });
            const outZipName = videoPath ? `colmap_video_dataset_${runId}.zip` : `colmap_dataset_${runId}.zip`;
            const outZipPath = path.join(downloadsDir, outZipName);

            // Set chunked SSE streaming headers
            res.writeHead(200, {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache, no-transform',
              'Connection': 'keep-alive',
            });

            const pythonScript = path.join(process.cwd(), 'scripts', 'colmap_runner.py');
            const childEnv = {
              ...process.env,
              COLMAP_BIN: '/Users/ritikkumarshahi/colmap_bin/bin/colmap',
              PATH: `/Users/ritikkumarshahi/colmap_bin/bin:/opt/homebrew/bin:${process.env.PATH || ''}`
            };

            const pyArgs = [
              pythonScript,
              '--workspace', workspaceDir,
              '--scaling', scaling,
              '--matching', matching,
              '--camera-model', cameraModel,
              '--output-zip', outZipPath
            ];
            if (videoPath) {
              pyArgs.push('--video', videoPath, '--fps', fps);
            }

            const child = spawn('python3', pyArgs, { env: childEnv });

            let bufferRemainder = '';
            child.stdout.on('data', (data: Buffer) => {
              const text = bufferRemainder + data.toString('utf8');
              const lines = text.split('\n');
              bufferRemainder = lines.pop() || '';
              for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed) {
                  try {
                    const parsed = JSON.parse(trimmed);
                    if (parsed.type === 'done') {
                      parsed.downloadUrl = `/downloads/${outZipName}`;
                    }
                    res.write(`data: ${JSON.stringify(parsed)}\n\n`);
                  } catch (e) {
                    res.write(`data: ${JSON.stringify({ type: 'log', message: trimmed })}\n\n`);
                  }
                }
              }
            });

            child.stderr.on('data', (data: Buffer) => {
              const msg = data.toString('utf8').trim();
              if (msg) {
                res.write(`data: ${JSON.stringify({ type: 'log', message: `[stderr] ${msg}` })}\n\n`);
              }
            });

            child.on('close', (code: number) => {
              console.log(`[ColmapAPI] Pipeline completed with exit code ${code}`);
              if (code === 0) {
                res.write(`data: ${JSON.stringify({
                  type: 'complete',
                  success: true,
                  downloadUrl: `/downloads/${outZipName}`,
                  message: 'Dataset ready and downloaded.'
                })}\n\n`);
              } else {
                res.write(`data: ${JSON.stringify({
                  type: 'error',
                  message: `Pipeline encountered an error (exit code ${code})`
                })}\n\n`);
              }
              res.end();
              // Schedule workspace cleanup
              setTimeout(() => {
                try { fs.rmSync(workspaceDir, { recursive: true, force: true }); } catch (e) {}
              }, 10 * 60 * 1000);
            });

            return;
          } catch (err: any) {
            console.error('[ColmapAPI] Error:', err);
            res.statusCode = 500;
            res.end(JSON.stringify({ error: err.message || String(err) }));
            return;
          }
        }

        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), incrementalApiPlugin()],
  server: {
    port: 3000,
    fs: {
      strict: false,
    },
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['@webgpu/types'],
  },
});
