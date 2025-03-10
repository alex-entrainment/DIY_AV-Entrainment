# Description of Available Tags 

### Available Tags: 
---
1. <schedule></schedule> - Contains full gnaural schedule 
2. <gnauralfile_version> - Indicates gnaural file version
3. <gnaural_version> - Indicates gnaural version
4. <date> - indicates date 
5. <title> - indicates schedule title
6. <totaltime> - indicates total run time 
7. <voicecount> - indicates the # of different "voices" in the schedule 
8. <totalentrycount> - indicates total # of entries in the schedule 
9. <loops> - # of times the schedule loops 
10. <overallvolume_left> - vol left 
11. <overallvolume_right> - vol right 
12. <stereoswap> - swaps stereo channels (usually 0) 
13. <graphview> - enables gnaural software graph view when loaded 
15. <voice> - contains the parameters for a particular voice across the schedule 
    - <description> - describes the voice 
    - <id> - Voice ID
    - <type> - Kind of "voice" - available are:
        0 Binaural Beat
        1 Pink Noise
        2 Audio File
        3 Isochronic Tone
        4 Alternate Isochronic Tone 
        5 Water Drops
        6 Rain
    - <voice_state> - Whether the "view" box is ticked on the gnaural interface (1)
    - <voice_hide> -  Whether the "view" box is ticked on the gnaural interface (0)
    - <voice_mute> - Whether the "mute" box is ticked on the gnaural interface
    - <voice_mono> - Is the voice mono or stereo
    - <entrycount> - Number of entries for the voice
    - <entries> - contains individual <entry> tags, should contain the <entrycount> number of <entry>s 
    - <entry> - Defines parameters for an entry within a voice, parameters include: 
        - parent="<id of the voice>"
        - duration="<duration in seconds>"
        - volume_left="<volume, 0.0-1.0>" 
        - volume_right="<volume, 0.0-1.0>"
        - beatfreq="<beatfrequency for the entry>"
        - basefreq="<base frequency used>" 
        - state="<0>"
    - 

        
