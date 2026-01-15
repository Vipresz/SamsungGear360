// gear360_mjpeg.cpp
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
    }
    #include <iostream>
    
    int main() {
        const char* url = "http://192.168.43.1:7679/livestream_high.avi";
    
        avformat_network_init();
    
        AVFormatContext* fmt = nullptr;
        AVDictionary* opts = nullptr;
    
        // Lower latency-ish options
        av_dict_set(&opts, "fflags", "nobuffer", 0);
        av_dict_set(&opts, "flags", "low_delay", 0);
        av_dict_set(&opts, "analyzeduration", "0", 0);
        av_dict_set(&opts, "probesize", "32", 0);
    
        if (avformat_open_input(&fmt, url, nullptr, &opts) < 0) {
            std::cerr << "Failed to open input\n";
            return 1;
        }
        av_dict_free(&opts);
    
        if (avformat_find_stream_info(fmt, nullptr) < 0) {
            std::cerr << "Failed to find stream info\n";
            return 1;
        }
    
        int videoStream = -1;
        for (unsigned i = 0; i < fmt->nb_streams; i++) {
            if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStream = (int)i;
                break;
            }
        }
        if (videoStream < 0) {
            std::cerr << "No video stream found\n";
            return 1;
        }
    
        const AVCodecParameters* par = fmt->streams[videoStream]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(par->codec_id);
        if (!codec) {
            std::cerr << "Decoder not found\n";
            return 1;
        }
    
        AVCodecContext* dec = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(dec, par);
    
        if (avcodec_open2(dec, codec, nullptr) < 0) {
            std::cerr << "Failed to open decoder\n";
            return 1;
        }
    
        AVPacket* pkt = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
    
        // Convert to BGR24 (optional; useful if you want cv::Mat)
        SwsContext* sws = sws_getContext(
            dec->width, dec->height, dec->pix_fmt,
            dec->width, dec->height, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
    
        int bgrStride[4] = { dec->width * 3, 0, 0, 0 };
        std::vector<uint8_t> bgr(dec->width * dec->height * 3);
        uint8_t* bgrData[4] = { bgr.data(), nullptr, nullptr, nullptr };
    
        long long frameCount = 0;
    
        while (av_read_frame(fmt, pkt) >= 0) {
            if (pkt->stream_index != videoStream) {
                av_packet_unref(pkt);
                continue;
            }
    
            if (avcodec_send_packet(dec, pkt) == 0) {
                while (avcodec_receive_frame(dec, frame) == 0) {
                    sws_scale(sws, frame->data, frame->linesize, 0, dec->height, bgrData, bgrStride);
    
                    // bgr now contains one frame (BGR24)
                    // TODO: feed to OpenCV / your pipeline
                    frameCount++;
                    if (frameCount % 25 == 0) {
                        std::cout << "Decoded frames: " << frameCount << "\n";
                    }
                }
            }
    
            av_packet_unref(pkt);
        }
    
        sws_freeContext(sws);
        av_frame_free(&frame);
        av_packet_free(&pkt);
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
        return 0;
    }