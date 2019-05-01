
                var reader = new FileReader();
                reader.onloadend = function(evt) {

                    /*var wavesurfer = WaveSurfer.create({
                        container: '#waveform-spectrogram',
                        plugins: [
                            WaveSurfer.spectrogram.create({
                                wavesurfer: wavesurfer,
                                container: "#waveform",
                                labels: true
                            })
                        ]
                    });*/
                    wavesurfer.load("http://ia902606.us.archive.org/35/items/shortpoetry_047_librivox/song_cjrg_teasdale_64kb.mp3");

                    /*var sound = new Howl({
                        src: [evt.target.result],
                        sprite: {
                            blast: [0, 3000],
                            laser: [4000, 1000],
                            winner: [6000, 5000]
                          }
                    });
                      
                    sound.play('laser');
                    */
                }
                reader.readAsDataURL(document.getElementById("audioFile").files[0])





                
                
    <script src="https://cdnjs.cloudflare.com/ajax/libs/howler/2.1.2/howler.js"></script>
    <script>
        /*
        var sound = new Howl({
            src: ['sounds.webm'],
            sprite: {
                blast: [0, 3000],
                laser: [4000, 1000],
                winner: [6000, 5000]
            }
            });
            
            // Shoot the laser!
            sound.play('laser');
        */
    </script>