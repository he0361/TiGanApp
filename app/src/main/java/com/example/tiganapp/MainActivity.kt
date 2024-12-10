package com.example.tiganapp
import org.java_websocket.client.WebSocketClient
import org.java_websocket.handshake.ServerHandshake
import java.net.URI
import java.nio.ByteBuffer
import android.util.Log
import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.tiganapp.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.widget.Toast
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.PermissionChecker
import java.io.ByteArrayOutputStream
import java.text.SimpleDateFormat
import java.util.Locale
import java.net.HttpURLConnection
import java.net.URL
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Environment
import androidx.annotation.RequiresApi
import java.io.File
import java.io.OutputStream
import java.net.Socket
import java.net.InetSocketAddress
class MainActivity : AppCompatActivity() {
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>,
        grantResults:
        IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
   /*
    // LuminosityAnalyzer 类实现了 ImageAnalysis.Analyzer 接口，用于分析图像的亮度（光照）信息
    private class LuminosityAnalyzer(private val listener: LumaListener) : ImageAnalysis.Analyzer {
    //定义了一个扩展函数 toByteArray()，扩展了 ByteBuffer 类型。通过这个函数，可以将 ByteBuffer 中的数据转化为一个 ByteArray 数组，便于后续的处理
        private fun ByteBuffer.toByteArray(): ByteArray {
            //重置缓冲区的 position，将其指针重置为 0。这样可以确保从缓冲区的起始位置开始读取数据。
            rewind()    // Rewind the buffer to zero
        //remaining()：返回当前缓冲区中剩余的字节数。这里用它来创建一个新的 ByteArray，大小为缓冲区中剩余的数据长度。
            val data = ByteArray(remaining())
        //这会将缓冲区中的字节数据复制到新创建的字节数组 data 中。
            get(data)   // Copy the buffer into a byte array
            return data // Return the byte array
        }
//该方法会在每次分析图像时被调用。image 是一个 ImageProxy 对象，代表捕获的图像数据。
        override fun analyze(image: ImageProxy) {
//从 image 中获取图像的第一个平面（planes[0]），并从中提取出数据缓冲区（buffer）。
//image.planes 是一个 Image.Plane 数组，通常图像数据会分成多个平面进行存储，这里取第一个平面（通常是Y平面，即亮度数据）。
            val buffer = image.planes[0].buffer
//调用刚才定义的 toByteArray() 扩展函数，将缓冲区中的数据转换为字节数组 data。
            val data = buffer.toByteArray()
            val pixels = data.map { it.toInt() and 0xFF }
    //计算亮度值（luma）的平均值。pixels 是一个整数列表，包含了图像中所有像素的亮度值。average() 方法会计算该列表的平均值，代表图像的平均亮度。
            val luma = pixels.average()
            listener(luma)
//将计算出的 luma（图像的平均亮度）传递给 listener，调用传入的回调函数，将亮度值返回出去，可能用于界面显示或进一步处理。
            image.close()
        }
    }
*/
    /*  对图片每一帧进行处理
   private class FrameProcessor(private val listener: (ByteArray) -> Unit) : ImageAnalysis.Analyzer {
       // 将 ByteBuffer 转换为字节数组
       private fun ByteBuffer.toByteArray(): ByteArray {
           rewind()    // 重置缓冲区的读指针
           val data = ByteArray(remaining())  // 创建一个与缓冲区剩余数据大小相等的字节数组
           get(data)   // 将缓冲区的数据复制到字节数组中
           return data // 返回字节数组
       }
       @OptIn(ExperimentalGetImage::class)
       override fun analyze(image: ImageProxy) {
           // 获取图像的 YUV 数据
           val imageData = image.image

           if (imageData != null) {
               // 将图像数据转换为字节数组
               val imageByteArray = imageToByteArray(imageData)

               // 将图像字节数据传递给 Python 代码块进行处理
               listener(imageByteArray)

           } else {
               Log.e(TAG, "Image data is null") // 如果获取不到图像数据，打印错误日志
           }

           // 确保在处理完后关闭 ImageProxy，避免内存泄漏
           image.close()
       }
       // 将 YUV 图像数据转换为字节数组
       private fun imageToByteArray(image: Image): ByteArray {
           val buffer: ByteBuffer = image.planes[0].buffer
           val bytes = ByteArray(buffer.remaining())
           buffer.get(bytes) // 将缓冲区的数据读取到字节数组中
           return bytes
       }
   }
*/
   @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
   class VideoStreamProcessor(private val listener: (ByteArray) -> Unit) : ImageAnalysis.Analyzer {

       private val TAG = "VideoStreamProcessor"

       // MediaCodec用于视频编码
       private var mediaCodec: MediaCodec? = null

       // 存储视频编码的格式（如分辨率、比特率、帧率等）
       private var mediaFormat: MediaFormat? = null

       // 用于写入视频文件的MediaMuxer实例
       private var mediaMuxer: MediaMuxer? = null

       // 视频轨道索引，用于指定媒体文件中的视频数据流
       private var videoTrackIndex: Int = -1

       // 视频是否正在编码
       private var isEncoding = false

       // 构造函数，初始化MediaCodec和网络连接
       init {
           try {
               // 创建视频编码格式，指定视频的MIME类型（H.264）和分辨率（640x480）
               Log.d(TAG, "我开始设置编码格式了")
               mediaFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, 640, 480)
               Log.d(TAG, "我开始设置颜色格式了")
               mediaFormat?.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar)  // 设置颜色格式
               Log.d(TAG, "我开始设置比特率了")
               mediaFormat?.setInteger(MediaFormat.KEY_BIT_RATE, 1000000)  // 设置视频编码的比特率
               Log.d(TAG, "我开始设置帧率了")
               mediaFormat?.setInteger(MediaFormat.KEY_FRAME_RATE, 30)  // 设置视频的帧率
               Log.d(TAG, "我开始设置帧间隔了")
               mediaFormat?.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 5)  // 设置关键帧的间隔
               // 创建MediaCodec实例，指定H.264编码器
               Log.d(TAG, "我开始设置264编码器了")
               mediaCodec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
               // 检查创建的mediaCodec是否为空
               if (mediaCodec == null) {
                   Log.e(TAG, "Failed to create MediaCodec encoder for H.264 format")
               } else {
                   Log.d(TAG, "Successfully created MediaCodec encoder for H.264 format")
               }
               Log.d(TAG, "我开始配置MediaCodec")
               // 配置MediaCodec，设置编码器的输出目标为空（将直接获取编码后的数据）
               mediaCodec?.configure(mediaFormat,null,null, MediaCodec.CONFIGURE_FLAG_ENCODE)
               Log.d(TAG, "我开始进入执行状态了")
               // 启动MediaCodec，使其进入“Executing”状态
               mediaCodec?.start()
               Log.d(TAG, "我开始初始化了")
               Log.d(TAG, "我开始初始化了")
               Log.d(TAG, "我开始初始化了")
               // 假设启动成功，设置isEncoding为true
               isEncoding = true
               Log.d(TAG, "我初始化成功了")
               Log.d(TAG, "我初始化成功了")
               Log.d(TAG, "我初始化成功了")

               Log.d(TAG, "MediaCodec initialized and started.")  // 输出日志，确认MediaCodec已初始化并开始

           } catch (e: Exception) {
               // 如果初始化过程中发生错误，打印错误日志
               Log.e(TAG, "Error initializing MediaCodec: ${e.message}")
               Log.e(TAG, "Stack trace: ${Log.getStackTraceString(e)}") // 打印完整的堆栈信息
               isEncoding = false  // 如果初始化失败，设置isEncoding为false
           }
       }



       // ImageAnalysis的analyze方法，每捕获一帧图像就会调用此方法
       @ExperimentalGetImage
       override fun analyze(image: ImageProxy) {
           // 如果 MediaCodec 处于执行状态，才进行编码
           if (!isEncoding) {
               Log.e(TAG, "MediaCodec is not in executing state, skipping frame encoding")
               image.close()
               return
           }

           val imageData = image.image
           // 如果成功获取到图像数据
           if (imageData != null) {
               // 打印图像的基本信息，以验证获取到图像
               Log.d(TAG, "Captured image: width = ${imageData.width}, height = ${imageData.height}")
               // 将图像数据转换为字节数组
               val imageByteArray = imageToByteArray(imageData)
               // 打印字节数组的大小，验证是否成功转换图像数据
               Log.d(TAG, "Image byte array size: ${imageByteArray.size} bytes")
               // 将图像帧数据编码成视频流
               val encodedVideoStream = encodeFrameToVideoStream(imageByteArray)

               // 如果成功编码，写入视频流数据
               if (encodedVideoStream.isNotEmpty()) {
                   Log.d(TAG, "Encoded video stream size: ${encodedVideoStream.size} bytes")
                   listener(encodedVideoStream)
               } else {
                   // 编码失败时，输出错误日志
                   Log.e(TAG, "Failed to encode frame")
               }
           } else {
               // 如果未能获取到图像数据，输出错误日志
               Log.e(TAG, "Image data is null")
           }

           // 关闭ImageProxy，释放资源
           image.close()
       }

       // 将Image中的YUV图像数据转换为字节数组
       private fun imageToByteArray(image: Image): ByteArray {
           val buffer: ByteBuffer = image.planes[0].buffer
           val bytes = ByteArray(buffer.remaining())
           buffer.get(bytes)
           return bytes
       }

       // 对图像帧进行编码，将其转换为视频流（例如H.264格式的视频流）
       private fun encodeFrameToVideoStream(frameData: ByteArray): ByteArray {
           try {
               // 获取输入缓冲区的索引
               val inputBufferIndex = mediaCodec?.dequeueInputBuffer(10000) ?: -1
               if (inputBufferIndex >= 0) {
                   val inputBuffer = mediaCodec?.getInputBuffer(inputBufferIndex)
                   inputBuffer?.clear()
                   inputBuffer?.put(frameData)

                   // 提交数据给编码器进行编码
                   mediaCodec?.queueInputBuffer(inputBufferIndex, 0, frameData.size, System.nanoTime() / 1000, 0)
               }

               // 创建BufferInfo实例，用于存储编码后数据的相关信息
               val bufferInfo = MediaCodec.BufferInfo()
               val outputBufferIndex = mediaCodec?.dequeueOutputBuffer(bufferInfo, 10000) ?: -1
               if (outputBufferIndex >= 0) {
                   val outputBuffer = mediaCodec?.getOutputBuffer(outputBufferIndex)
                   val outputData = ByteArray(bufferInfo.size)
                   outputBuffer?.get(outputData)

                   // 释放输出缓冲区
                   mediaCodec?.releaseOutputBuffer(outputBufferIndex, false)

                   return outputData
               }
           } catch (e: Exception) {
               Log.e(TAG, "Error encoding frame to video stream: ${e.message}")
           }
           return ByteArray(0)
       }

       // 释放资源，停止MediaCodec
       fun release() {
           // 确保MediaCodec处于执行状态
           if (isEncoding) {
               mediaCodec?.stop()
               mediaCodec?.release()
               isEncoding = false
               Log.d(TAG, "MediaCodec released successfully.")
           }
       }
   }


    private lateinit var viewBinding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private lateinit var cameraExecutor: ExecutorService
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Set up the listeners for take photo and video capture buttons
        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }
        viewBinding.videoCaptureButton.setOnClickListener { captureVideo() }
        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    private fun startCamera() {
        // 获取 CameraX 的相机提供者实例，返回一个 CameraProviderFuture 对象，用于异步操作
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        // 在获取到相机提供者实例后执行以下操作
        cameraProviderFuture.addListener({
            // 获取实际的 CameraProvider 实例，CameraProvider 用于管理所有的相机相关操作
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .build()
            // 创建图像分析用例，用于分析相机帧数据
//            val imageAnalyzer = ImageAnalysis.Builder()
//                .build()  // 构建图像分析对象
//                //// 设置图像分析的分析器（LuminosityAnalyzer），并指定执行线程池为 cameraExecutor
//                .also {
//                    it.setAnalyzer(cameraExecutor, LuminosityAnalyzer { luma ->
//                        Log.d(TAG, "Average luminosity: $luma")
//                    })
//                }
            // 设置图像分析器
            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, VideoStreamProcessor { videoStream ->
                        // 将编码后的视频流发送到Python服务器进行处理
                        sendDataToPythonServer(videoStream)
                        Log.d(TAG, "体感APP->: $videoStream")
                    })
                }
            // 打开后置摄像头
//            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            try {
                // 在重新绑定新的相机用例之前，解除所有之前绑定的用例
                cameraProvider.unbindAll()
                // 将相机的预览用例与生命周期绑定
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))//使用 getMainExecutor() 获取主线程的执行器，确保 addListener 中的代码在主线程中运行，因为 UI 更新需要在主线程中进行。
    }
    // 拍照函数，用于捕获并保存照片
    private fun takePhoto() {
        // 获取一个可修改的 ImageCapture 实例，确保相机已经准备好
        // 如果 imageCapture 为 null，则直接返回，不执行拍照操作
        val imageCapture = imageCapture ?: return
        // 创建拍照的文件名，并为该文件在 MediaStore 中创建条目
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())//System.currentTimeMillis() 是一个返回当前时间，表示文件名是当前的时间
        val contentValues = ContentValues().apply {
            // 设置文件名和 MIME 类型
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")//用来指定图片保存的相对路径（在这里，指定了 "Pictures/CameraX-Image" 文件夹）。
            }
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver, // 通过它，应用可以将图片存储到公共目录下，如“相册”文件夹，而不需要直接操作文件系统。
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,// MediaStore提供的一个 URI，指向设备外部存储的图片目录
                contentValues) //是用来设置图片的名称、类型、路径等信息的
            .build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        // 设置图像捕获监听器，该监听器在拍照后被触发
        imageCapture.takePicture(  //outputOptions: 确定文件保存路径和元数据。 executor: 指定在哪个线程中执行回调，这里使用 ContextCompat.getMainExecutor(this) 来确保回调在主线程执行，因为 UI 操作需要在主线程进行。
            outputOptions, // 输出文件选项，指定文件保存路径和元数据
            ContextCompat.getMainExecutor(this), // 使用主线程的执行器来执行回调（因为 UI 操作必须在主线程中执行）
            object : ImageCapture.OnImageSavedCallback {
                // 如果拍照过程发生错误，调用 onError 方法
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }
                // 拍照成功后，调用 onImageSaved 方法
                override fun onImageSaved(output: ImageCapture.OutputFileResults){
                    // 成功捕获并保存照片，获取保存的照片 URI
                    val msg = "Photo capture succeeded: ${output.savedUri}"
                    // 显示成功的 Toast 提示
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    // 打印成功的日志消息
                    Log.d(TAG, msg)
                }
            }
        )
    }

    // Implements VideoCapture use case, including start and stop capturing.
    private fun captureVideo() {
        val videoCapture = this.videoCapture ?: return

        viewBinding.videoCaptureButton.isEnabled = false

        val curRecording = recording
        if (curRecording != null) {
            // Stop the current recording session.
            curRecording.stop()
            recording = null
            return
        }

        // create and start a new recording session
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/CameraX-Video")
            }
        }

        val mediaStoreOutputOptions = MediaStoreOutputOptions
            .Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues)
            .build()
        recording = videoCapture.output
            .prepareRecording(this, mediaStoreOutputOptions)
            .apply {
                if (PermissionChecker.checkSelfPermission(this@MainActivity,
                        Manifest.permission.RECORD_AUDIO) ==
                    PermissionChecker.PERMISSION_GRANTED)
                {
                    withAudioEnabled()
                }
            }
            .start(ContextCompat.getMainExecutor(this)) { recordEvent ->
                when(recordEvent) {
                    is VideoRecordEvent.Start -> {
                        viewBinding.videoCaptureButton.apply {
                            text = getString(R.string.stop_capture)
                            isEnabled = true
                        }
                    }
                    is VideoRecordEvent.Finalize -> {
                        if (!recordEvent.hasError()) {
                            val msg = "Video capture succeeded: " +
                                    "${recordEvent.outputResults.outputUri}"
                            Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT)
                                .show()
                            Log.d(TAG, msg)
                        } else {
                            recording?.close()
                            recording = null
                            Log.e(TAG, "Video capture ends with error: " +
                                    "${recordEvent.error}")
                        }
                        viewBinding.videoCaptureButton.apply {
                            text = getString(R.string.start_capture)
                            isEnabled = true
                        }
                    }
                }
            }
    }
    // 启动相机的函数


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }
    // Python WebSocket 服务器地址
    val serverURI = URI("ws://10.0.2.2:8765") // 模拟器连接本地机器
    private fun sendDataToPythonServer(data: ByteArray) {
        // 创建 WebSocket 客户端
        val client = object : WebSocketClient(serverURI) {
            override fun onOpen(handshakedata: ServerHandshake?) {
                Log.d(TAG, "WebSocket connected")
                // 连接成功后，发送视频流数据
                send(data)
            }
        //这个函数在 WebSocket 客户端接收到从服务器发送的消息时被调用
            override fun onMessage(message: String?) {
                Log.d(TAG, "Received message from server: $message")
            }
       //这个函数在 WebSocket 连接关闭时被调用
            override fun onClose(code: Int, reason: String?, remote: Boolean) {
                Log.d(TAG, "WebSocket closed: $reason")
            }

            override fun onError(ex: Exception?) {
                Log.e(TAG, "WebSocket error", ex)
                ex?.printStackTrace()
            }
            // 确保调用 connect() 方法发起连接
        }
        client.connect()
//
//        // 在后台线程中启动 WebSocket 客户端
//        Thread {
//            try {
//                client.connectBlocking() // 阻塞直到 WebSocket 连接建立
//            } catch (e: Exception) {
//                Log.e(TAG, "Error connecting to WebSocket server", e)
//            }
//        }.start()
    }

    //将数据发送到python
    private fun sendToPython(imageByteArray: ByteArray) {
        // 创建一个新的线程来发送图像数据到 Python
        Thread {
            try {
                // 创建一个 URL 对象，指向 Python 服务器的 API
                val url = URL("http://your-python-server-address/api/process_image")

                // 创建一个 HTTP 连接对象
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/octet-stream")
                connection.doOutput = true

                // 发送图像字节数据到 Python 服务器
                connection.outputStream.write(imageByteArray)

                // 读取响应
                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    Log.d(TAG, "Image sent to Python successfully")
                } else {
                    Log.e(TAG, "Failed to send image to Python: $responseCode")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error sending image to Python", e)
            }
        }.start()
    }

}
