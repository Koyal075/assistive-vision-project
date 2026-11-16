package com.example.assistivevision

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.assistivevision.ui.theme.AssistiveVisionTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.min
import android.speech.tts.TextToSpeech
import java.util.Locale
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions

// ---- Tuning knobs (change these to adjust behaviour) ----
const val TFLITE_THREADS = 2             // most phones have 2 fast cores; try 2 first, then 4
const val ROUND_INTERVAL_MS = 400L       // minimum time between two detection rounds
const val AUTO_OCR_ENABLED = true        // set to false to read text ONLY when you tap the screen
const val OCR_INTERVAL_MS = 3000L        // how often automatic OCR runs
const val AUTO_TEXT_MIN_AREA = 0.02f     // text must fill >= 2% of the frame to be read automatically
const val TEXT_REPEAT_MS = 30_000L       // the same text is not re-read for this long
const val OBJECT_REPEAT_MS = 5_000L      // the same object alert is not repeated for this long
const val MIN_GAP_BETWEEN_ALERTS_MS = 1500L

data class Detection(
    val label: String,
    val confidence: Float,
    val left: Float,   // normalized 0..1, relative to the ORIGINAL camera frame
    val top: Float,
    val right: Float,
    val bottom: Float
)

data class LetterboxInfo(
    val scale: Float,
    val padLeft: Float,
    val padTop: Float,
    val origWidth: Int,
    val origHeight: Int
)

// A candidate for the single spoken alert this round: an object OR a piece of text
data class SpokenCandidate(val phrase: String, val area: Float, val key: String, val repeatMs: Long)

// One block of text found by OCR, with how much of the frame it fills (0..1)
data class OcrItem(val text: String, val area: Float)

class MainActivity : ComponentActivity() {

    private var hasCameraPermission = false

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        recreate()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        hasCameraPermission = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasCameraPermission) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            AssistiveVisionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { _ ->
                    if (hasCameraPermission) {
                        CameraPreviewScreen(modifier = Modifier.fillMaxSize())
                    } else {
                        Text(
                            text = "Camera permission is required.",
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                }
            }
        }
    }
}

fun loadModelFile(context: Context, fileName: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(fileName)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

// Trims each line and drops blank ones, so a stray empty line at the end of labels.txt
// can't make the label count disagree with the model.
fun loadLabels(context: Context): List<String> {
    return context.assets.open("labels.txt").bufferedReader().readLines()
        .map { it.trim() }
        .filter { it.isNotEmpty() }
}

// ---- Text helpers ----

// Collapse newlines/extra spaces so TTS reads the text as one smooth phrase.
fun cleanOcrText(raw: String): String = raw.replace(Regex("\\s+"), " ").trim()

// Reject OCR noise such as "|", "-", "I" or "~DS": at least 3 letters/digits,
// and at least half of the characters must be letters/digits.
fun looksLikeRealText(text: String): Boolean {
    val alnum = text.count { it.isLetterOrDigit() }
    return alnum >= 3 && alnum * 2 >= text.length
}

// Keep spoken text short: cut at a word boundary near maxChars.
fun shorten(text: String, maxChars: Int = 140): String {
    if (text.length <= maxChars) return text
    val cut = text.lastIndexOf(' ', maxChars)
    return text.substring(0, if (cut > 40) cut else maxChars)
}

// OCR returns slightly different text from run to run, so compare on a loose key:
// lowercase letters/digits only, first 16 characters.
fun textKey(text: String): String = text.lowercase().filter { it.isLetterOrDigit() }.take(16)

fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
    if (rotationDegrees == 0) return bitmap
    val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

// Resize WITHOUT distorting: scale to fit inside targetSize, pad the rest grey.
// This is only for MODEL INPUT, separate from how the frame is displayed on screen.
fun letterboxBitmap(bitmap: Bitmap, targetSize: Int): Pair<Bitmap, LetterboxInfo> {
    val origW = bitmap.width
    val origH = bitmap.height
    val scale = min(targetSize.toFloat() / origW, targetSize.toFloat() / origH)
    val newW = (origW * scale).toInt()
    val newH = (origH * scale).toInt()
    val padLeft = (targetSize - newW) / 2f
    val padTop = (targetSize - newH) / 2f

    val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
    val padded = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
    val canvas = android.graphics.Canvas(padded)
    canvas.drawColor(android.graphics.Color.rgb(114, 114, 114))
    canvas.drawBitmap(scaled, padLeft, padTop, null)

    return Pair(padded, LetterboxInfo(scale, padLeft, padTop, origW, origH))
}

fun nonMaxSuppression(detections: List<Detection>, iouThreshold: Float = 0.45f): List<Detection> {
    val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
    val result = mutableListOf<Detection>()
    while (sorted.isNotEmpty()) {
        val best = sorted.removeAt(0)
        result.add(best)
        sorted.removeAll { other -> calculateIoU(best, other) > iouThreshold }
    }
    return result
}

fun calculateIoU(a: Detection, b: Detection): Float {
    val interLeft = max(a.left, b.left)
    val interTop = max(a.top, b.top)
    val interRight = min(a.right, b.right)
    val interBottom = min(a.bottom, b.bottom)
    val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
    val areaA = (a.right - a.left) * (a.bottom - a.top)
    val areaB = (b.right - b.left) * (b.bottom - b.top)
    return if (areaA + areaB - interArea <= 0f) 0f else interArea / (areaA + areaB - interArea)
}

enum class Severity { CRITICAL, WARNING, SAFE }

data class ClassifiedDetection(
    val detection: Detection,
    val location: String,
    val severity: Severity,
    val reason: String,
    val area: Float
)

fun classifyDetection(det: Detection): ClassifiedDetection {
    val centerX = (det.left + det.right) / 2f
    val centerY = (det.top + det.bottom) / 2f
    val boxArea = (det.right - det.left) * (det.bottom - det.top)

    val horizontal = when {
        centerX < 0.33f -> "left"
        centerX > 0.66f -> "right"
        else -> "center"
    }
    val vertical = when {
        centerY < 0.33f -> "upper"
        centerY > 0.66f -> "ground level"
        else -> "mid-level"
    }
    val location = "$vertical $horizontal"

    val inWalkingPath = centerX in 0.25f..0.75f
    val isVeryClose = boxArea > 0.20f
    val isSomewhatClose = boxArea > 0.12f

    val severity: Severity
    val reason: String
    when {
        inWalkingPath && isVeryClose -> { severity = Severity.CRITICAL; reason = "Close obstruction directly in walking path" }
        isVeryClose -> { severity = Severity.WARNING; reason = "Object very close nearby" }
        inWalkingPath && isSomewhatClose -> { severity = Severity.WARNING; reason = "Object approaching in walking path" }
        else -> { severity = Severity.SAFE; reason = "Distant or non-obstructive" }
    }
    return ClassifiedDetection(det, location, severity, reason, boxArea)
}

@Composable
fun CameraPreviewScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current

    var detections by remember { mutableStateOf(listOf<Detection>()) }
    var frameSize by remember { mutableStateOf(Size(1f, 1f)) }
    var caption by remember { mutableStateOf("") }   // text found by OCR, shown at the bottom (objects are never shown here)
    val captionTime = remember { mutableStateOf(0L) }

    val labels = remember { loadLabels(context) }
    val lastSpokenTime = remember { mutableStateOf(0L) }
    val spokenAt = remember { mutableMapOf<String, Long>() }   // alert key -> last time it was spoken
    val pendingKey = remember { mutableStateOf("") }
    val pendingStreak = remember { mutableStateOf(0) }
    val readRequest = remember { mutableStateOf(false) }       // set to true when the user taps the screen

    // All camera analysis (model + OCR preparation) runs on this ONE background thread,
    // so the screen and touch handling stay responsive while the model is busy.
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }

    val tts = remember {
        var ttsInstance: TextToSpeech? = null
        ttsInstance = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) ttsInstance?.language = Locale.US
        }
        ttsInstance
    }

    Box(modifier = modifier) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                // FILL_CENTER = image covers the whole screen (may crop edges)
                previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

                val options = Interpreter.Options().apply { setNumThreads(TFLITE_THREADS) }
                val interpreter = Interpreter(loadModelFile(ctx, "best.tflite"), options)
                val textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                val inputShape = interpreter.getInputTensor(0).shape()
                val modelInputSize = inputShape[2]
                val outputShape = interpreter.getOutputTensor(0).shape()
                val numAttrs = outputShape[1]
                val numBoxes = outputShape[2]
                val numClasses = numAttrs - 4
                Log.d("ASSISTIVE_VISION", "Model: input=$modelInputSize classes=$numClasses boxes=$numBoxes threads=$TFLITE_THREADS")
                if (labels.size != numClasses) {
                    Log.e("ASSISTIVE_VISION", "MISMATCH: labels.txt has ${labels.size} lines but the model has $numClasses classes")
                }

                // Buffers are created ONCE and reused every round (no per-frame allocation).
                val inputBuffer = ByteBuffer
                    .allocateDirect(1 * 3 * modelInputSize * modelInputSize * 4)
                    .order(ByteOrder.nativeOrder())
                val pixels = IntArray(modelInputSize * modelInputSize)
                val output = Array(1) { Array(numAttrs) { FloatArray(numBoxes) } }

                // OCR results come back on the main thread but are read on the analysis thread,
                // so they live in thread-safe holders.
                val ocrBlocksRef = AtomicReference<List<OcrItem>>(emptyList())   // text seen in two OCR runs in a row
                val prevOcrKeysRef = AtomicReference<Set<String>>(emptySet())    // text keys from the previous OCR run
                val ocrBusy = AtomicBoolean(false)

                // These are only touched on the analysis thread.
                var lastOcrTime = 0L
                var lastRoundStart = 0L
                var loggedCoordScale = false

                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy: ImageProxy ->
                        try {
                            // ================= TAP TO READ =================
                            // The user tapped the screen: run OCR right now on this frame and read
                            // everything found (spoken and shown on screen). Stays silent if no text is found.
                            if (readRequest.value) {
                                readRequest.value = false
                                try {
                                    val tapBitmap = rotateBitmap(imageProxy.toBitmap(), imageProxy.imageInfo.rotationDegrees)
                                    textRecognizer.process(InputImage.fromBitmap(tapBitmap, 0))
                                        .addOnSuccessListener { visionText ->
                                            val blocks = visionText.textBlocks
                                                .map { cleanOcrText(it.text) }
                                                .filter { looksLikeRealText(it) }
                                            Log.d("ASSISTIVE_VISION", "TAP READ: ${blocks.size} text blocks: $blocks")
                                            // Silent when nothing is found; only real text is spoken and shown.
                                            if (blocks.isNotEmpty()) {
                                                val toRead = blocks.take(8)
                                                toRead.forEachIndexed { i, t ->
                                                    tts?.speak(
                                                        shorten(t, 250),
                                                        if (i == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD,
                                                        null,
                                                        "read_$i"
                                                    )
                                                }
                                                caption = toRead.joinToString("\n") { shorten(it, 250) }
                                                captionTime.value = System.currentTimeMillis()
                                            }
                                        }
                                        .addOnFailureListener { e ->
                                            Log.e("ASSISTIVE_VISION", "Tap read OCR error: ${e.message}", e)
                                        }
                                } catch (e: Exception) {
                                    Log.e("ASSISTIVE_VISION", "Tap read error: ${e.message}", e)
                                }
                            }

                            // ================= DETECTION ROUND (at most one per ROUND_INTERVAL_MS) =================
                            val roundStart = System.currentTimeMillis()
                            if (roundStart - lastRoundStart >= ROUND_INTERVAL_MS) {
                                lastRoundStart = roundStart

                                val rawBitmap = imageProxy.toBitmap()
                                val rotated = rotateBitmap(rawBitmap, imageProxy.imageInfo.rotationDegrees)
                                frameSize = Size(rotated.width.toFloat(), rotated.height.toFloat())

                                // ---- Automatic OCR: never overlaps with itself ----
                                if (AUTO_OCR_ENABLED && !ocrBusy.get() && roundStart - lastOcrTime > OCR_INTERVAL_MS) {
                                    ocrBusy.set(true)
                                    lastOcrTime = roundStart
                                    textRecognizer.process(InputImage.fromBitmap(rotated, 0))
                                        .addOnSuccessListener { visionText ->
                                            val frameArea = (rotated.width * rotated.height).toFloat()
                                            val items = visionText.textBlocks.mapNotNull { block ->
                                                val box = block.boundingBox ?: return@mapNotNull null
                                                val text = cleanOcrText(block.text)
                                                if (!looksLikeRealText(text)) return@mapNotNull null
                                                OcrItem(text, box.width() * box.height() / frameArea)
                                            }
                                            // Only trust text that was also seen in the previous OCR run;
                                            // one-off garbage from a single blurry frame is ignored.
                                            val previous = prevOcrKeysRef.get()
                                            ocrBlocksRef.set(items.filter { textKey(it.text) in previous })
                                            prevOcrKeysRef.set(items.map { textKey(it.text) }.toSet())
                                            Log.d(
                                                "ASSISTIVE_VISION",
                                                "OCR run: ${items.size} blocks, ${ocrBlocksRef.get().size} stable " +
                                                        items.joinToString(prefix = "[", postfix = "]") {
                                                            "'${shorten(it.text, 30)}' fills ${(it.area * 100).toInt()}%"
                                                        }
                                            )
                                        }
                                        .addOnFailureListener { e ->
                                            Log.e("ASSISTIVE_VISION", "OCR error: ${e.message}", e)
                                        }
                                        .addOnCompleteListener { ocrBusy.set(false) }
                                }

                                // ---- YOLO object detection ----
                                val (padded, lb) = letterboxBitmap(rotated, modelInputSize)
                                inputBuffer.clear()
                                padded.getPixels(pixels, 0, modelInputSize, 0, 0, modelInputSize, modelInputSize)
                                for (p in pixels) inputBuffer.putFloat(((p shr 16) and 0xFF) / 255.0f)
                                for (p in pixels) inputBuffer.putFloat(((p shr 8) and 0xFF) / 255.0f)
                                for (p in pixels) inputBuffer.putFloat((p and 0xFF) / 255.0f)
                                val t1 = System.nanoTime()

                                interpreter.run(inputBuffer, output)
                                val t2 = System.nanoTime()

                                // Box coordinates may be in pixels or normalized 0..1 depending on the
                                // export; sample a few boxes to tell which.
                                var maxSample = 0f
                                for (j in 0 until min(50, numBoxes)) {
                                    maxSample = max(maxSample, output[0][0][j])
                                    maxSample = max(maxSample, output[0][1][j])
                                }
                                val coordScale = if (maxSample <= 1.5f) modelInputSize.toFloat() else 1f
                                if (!loggedCoordScale) {
                                    Log.d("ASSISTIVE_VISION", "Box coord scale: maxSample=$maxSample -> coordScale=$coordScale")
                                    loggedCoordScale = true
                                }

                                val newDetections = mutableListOf<Detection>()
                                for (j in 0 until numBoxes) {
                                    val cxPix = output[0][0][j] * coordScale
                                    val cyPix = output[0][1][j] * coordScale
                                    val wPix = output[0][2][j] * coordScale
                                    val hPix = output[0][3][j] * coordScale

                                    var bestScore = 0f
                                    var bestClass = -1
                                    for (c in 0 until numClasses) {
                                        val score = output[0][4 + c][j]
                                        if (score > bestScore) { bestScore = score; bestClass = c }
                                    }

                                    if (bestScore > 0.55f && bestClass in labels.indices) {
                                        val cx = (cxPix - lb.padLeft) / lb.scale
                                        val cy = (cyPix - lb.padTop) / lb.scale
                                        val w = wPix / lb.scale
                                        val h = hPix / lb.scale

                                        val left = ((cx - w / 2f) / lb.origWidth).coerceIn(0f, 1f)
                                        val top = ((cy - h / 2f) / lb.origHeight).coerceIn(0f, 1f)
                                        val right = ((cx + w / 2f) / lb.origWidth).coerceIn(0f, 1f)
                                        val bottom = ((cy + h / 2f) / lb.origHeight).coerceIn(0f, 1f)

                                        newDetections.add(Detection(labels[bestClass], bestScore, left, top, right, bottom))
                                    }
                                }

                                detections = nonMaxSuppression(newDetections)

                                // ================= WHAT TO SPEAK =================
                                // Objects and text compete on how much of the frame they fill (a bigger
                                // area means closer). Only the single closest item is spoken.
                                val candidates = mutableListOf<SpokenCandidate>()

                                detections.map { classifyDetection(it) }.forEach { c ->
                                    if (c.severity == Severity.CRITICAL) {
                                        candidates.add(
                                            SpokenCandidate(
                                                "${c.detection.label} very close, ${c.location}",
                                                c.area,
                                                "obj_${c.detection.label}",
                                                OBJECT_REPEAT_MS
                                            )
                                        )
                                    }
                                }
                                ocrBlocksRef.get().forEach { item ->
                                    if (item.area >= AUTO_TEXT_MIN_AREA) {
                                        candidates.add(
                                            SpokenCandidate(
                                                "Text: ${shorten(item.text)}",
                                                item.area,
                                                "text_${textKey(item.text)}",
                                                TEXT_REPEAT_MS
                                            )
                                        )
                                    }
                                }

                                val now = System.currentTimeMillis()
                                if (caption.isNotEmpty() && now - captionTime.value > 10_000L) caption = ""

                                val best = candidates.maxByOrNull { it.area }
                                if (best != null) {
                                    // The same candidate must win two rounds in a row, so a single
                                    // noisy frame can't trigger speech.
                                    if (best.key == pendingKey.value) {
                                        pendingStreak.value++
                                    } else {
                                        pendingKey.value = best.key
                                        pendingStreak.value = 1
                                    }
                                    val isText = best.key.startsWith("text_")
                                    val lastForKey = spokenAt[best.key] ?: 0L
                                    val repeatOk = now - lastForKey > best.repeatMs
                                    val gapOk = now - lastSpokenTime.value > MIN_GAP_BETWEEN_ALERTS_MS
                                    // Text never interrupts speech that is already playing (an object warning,
                                    // or a tap-to-read in progress); object warnings may interrupt anything.
                                    val textMustWait = isText && tts?.isSpeaking == true

                                    if (pendingStreak.value >= 2 && repeatOk && gapOk && !textMustWait) {
                                        tts?.speak(best.phrase, TextToSpeech.QUEUE_FLUSH, null, best.key)
                                        spokenAt[best.key] = now
                                        lastSpokenTime.value = now
                                        // Detected TEXT is also shown on screen; object alerts are spoken only.
                                        if (isText) {
                                            caption = best.phrase
                                            captionTime.value = now
                                        }
                                    }
                                } else {
                                    pendingKey.value = ""
                                    pendingStreak.value = 0
                                }

                                // Log only: inference time in ms, then each detected object with its confidence.
                                val inferMs = (t2 - t1) / 1_000_000
                                Log.d(
                                    "ASSISTIVE_VISION",
                                    "${inferMs}ms detected=" + detections.joinToString(prefix = "[", postfix = "]") {
                                        "${it.label} ${(it.confidence * 100).toInt()}%"
                                    }
                                )
                            }
                        } catch (e: Exception) {
                            Log.e("ASSISTIVE_VISION", "Analysis error: ${e.message}", e)
                        } finally {
                            imageProxy.close()
                        }
                    }

                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            }
        )

        // The overlay draws the bounding boxes with label + confidence, and also catches
        // taps anywhere on the screen (tap = read text aloud).
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    detectTapGestures(onTap = { readRequest.value = true })
                }
        ) {
            val canvasWidth = size.width
            val canvasHeight = size.height
            val frameW = frameSize.width
            val frameH = frameSize.height
            if (frameW <= 1f || frameH <= 1f) return@Canvas

            // COVER fit: matches FILL_CENTER, the frame fills the canvas and may be cropped.
            val coverScale = max(canvasWidth / frameW, canvasHeight / frameH)
            val displayedW = frameW * coverScale
            val displayedH = frameH * coverScale
            val offsetX = (canvasWidth - displayedW) / 2f
            val offsetY = (canvasHeight - displayedH) / 2f

            detections.forEach { det ->
                val left = offsetX + det.left * displayedW
                val top = offsetY + det.top * displayedH
                val right = offsetX + det.right * displayedW
                val bottom = offsetY + det.bottom * displayedH

                drawRect(
                    color = Color.Red,
                    topLeft = Offset(left, top),
                    size = Size(right - left, bottom - top),
                    style = Stroke(width = 4f)
                )
                drawContext.canvas.nativeCanvas.drawText(
                    "${det.label} ${(det.confidence * 100).toInt()}%",
                    left, (top - 10).coerceAtLeast(20f),
                    Paint().apply {
                        color = android.graphics.Color.RED
                        textSize = 36f
                        isFakeBoldText = true
                    }
                )
            }
        }

        // Shows detected text only (never object alerts); it is also read aloud.
        if (caption.isNotEmpty()) {
            Text(
                text = caption,
                color = Color.White,
                fontSize = 16.sp,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp)
                    .background(Color(0xCC000000))
                    .padding(10.dp)
            )
        }

        androidx.compose.runtime.DisposableEffect(Unit) {
            onDispose {
                analysisExecutor.shutdown()
                tts?.stop()
                tts?.shutdown()
            }
        }
    }
}