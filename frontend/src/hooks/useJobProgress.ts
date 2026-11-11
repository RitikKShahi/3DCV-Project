"use client";

import { useEffect, useRef, useState } from "react";
import { getJobStatus, getWsUrl } from "@/lib/api";
import type { JobResponse, ProgressWsMessage } from "@/lib/types";

export function useJobProgress(jobId: string | null | undefined, initialJob?: JobResponse | null) {
  const [job, setJob] = useState<JobResponse | null>(initialJob ?? null);
  const [previewRevision, setPreviewRevision] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (initialJob) setJob(initialJob);
  }, [initialJob]);

  useEffect(() => {
    if (!jobId) return;

    let stopped = false;
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    const refresh = async () => {
      try {
        const latest = await getJobStatus(jobId);
        if (!stopped) {
          setJob((prev) => {
            if (
              prev &&
              (latest.preview_batches.length > prev.preview_batches.length ||
                (latest.has_splat && !prev.has_splat))
            ) {
              setPreviewRevision((r) => r + 1);
            }
            return latest;
          });
        }
      } catch {
        // ignore transient poll failures
      }
    };

    refresh();

    // Connect WebSocket for instant updates
    try {
      const ws = new WebSocket(getWsUrl(jobId));
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as ProgressWsMessage;
          setJob((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              status: msg.status,
              stage: msg.stage,
              progress: msg.progress,
              message: msg.message,
              batch_index: msg.batch_index ?? prev.batch_index,
              total_batches: msg.total_batches ?? prev.total_batches,
              num_gaussians: msg.num_gaussians ?? prev.num_gaussians,
              num_cameras: msg.num_cameras ?? prev.num_cameras,
              psnr: msg.psnr ?? prev.psnr,
              loss: msg.loss ?? prev.loss,
              elapsed_sec: msg.elapsed_sec ?? prev.elapsed_sec,
              peak_memory_mb: msg.peak_memory_mb ?? prev.peak_memory_mb,
              error_message: msg.error_message ?? prev.error_message,
            };
          });
          if (msg.has_new_preview || msg.status === "completed") {
            setPreviewRevision((r) => r + 1);
            refresh();
          }
        } catch {
          // ignore malformed ws frame
        }
      };
    } catch {
      // rely on polling fallback
    }

    // Polling fallback guarantees state convergence even if WS drops
    pollTimer = setInterval(refresh, 2500);

    return () => {
      stopped = true;
      if (pollTimer) clearInterval(pollTimer);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [jobId]);

  return { job, previewRevision };
}

