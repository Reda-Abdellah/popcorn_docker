import os
import sys
import datetime
import csv
import numpy as np
import nibabel as nii
from string import Template
from PIL import Image
from skimage import filters
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label
from matplotlib import pyplot as plt
import numpy as np
import pickle

OUT_HEIGHT=217
DEFAULT_ALPHA=0.5


nii.Nifti1Header.quaternion_threshold = -8e-07
version = '1.0';
release_date = datetime.datetime.strptime("30-07-2021", "%d-%m-%Y").strftime("%d-%b-%Y")


#RGB
colormap={}
colormap[0] =[  0,    0,    0]
colormap[1] =[255,    0,    0]
colormap[2] =[  0,  255,    0]
colormap[3] =[  0,    0,  255]

##############

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_lesion_by_regions(fname, fname_crisp, fname_hemi, fname_lab, fname_lesion):

    juxtacortical_idx=3
    deepwhite_idx=2
    periventricular_idx=1
    cerebelar_idx=4
    medular_idx=5

    T1_img=nii.load(fname)
    crisp = nii.load(fname_crisp).get_data()
    hemi = nii.load(fname_hemi).get_data()
    lab = nii.load(fname_lab).get_data()
    lesion = nii.load(fname_lesion).get_data()

    ventricles=(lab==1)+(lab==2)
    cond1=(crisp==3)
    cond2=(lesion>0)
    structure=np.ones([5,5,5])
    cond3=binary_dilation( (lab>0), structure)
    wm_filled = binary_fill_holes( (cond1.astype('int')+ cond2.astype('int') + cond3.astype('int') )>0).astype('int')*6

    wm = (((crisp==3)+ (lesion>0))>0).astype('int')

    SE=np.zeros([5,5,5]);  # 3 mm distance
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if(((((i-3)**2)+((j-3)**2)+((k-3)**2))**0.5)<3):
                    SE[i,j,k]=1


    periventricular=wm_filled*binary_dilation(ventricles,SE)
    yuxtacortical=wm_filled-binary_erosion(wm_filled,SE)
    deep=abs(wm_filled-periventricular-yuxtacortical)>0

    cerebrum=(hemi==1)+(hemi==2)
    medular=hemi==5
    cerebelar=((hemi==4)+(hemi==3))*(crisp==3)
    #infratenttorial=medular+cerebelar

    regions=np.zeros(crisp.shape)
    ind=(cerebrum*yuxtacortical)>0
    regions[ind]=3
    ind=(cerebrum*deep)>0
    regions[ind]=2
    ind=(cerebrum*periventricular)>0
    regions[ind]=1
    ind=(cerebelar)>0
    regions[ind]=4
    ind=(medular)>0
    regions[ind]=5

    #result
    region_name= fname_crisp.replace('crisp','regions')
    wm_name= fname_crisp.replace('crisp','wmmap')
    nii.Nifti1Image(regions, T1_img.affine).to_filename(region_name)
    nii.Nifti1Image(wm, T1_img.affine).to_filename(wm_name)


    #clasification
    seg_labels, seg_num = label(lesion, return_num=True, connectivity=2)

    # Lesion analysis
    lesion2=np.zeros(lesion.shape)
    for i in range(1,seg_num+1):
        # Clasification
        ind=(seg_labels==i)
        votes=regions[ind]
        #periventicular
        if( (votes==periventricular_idx).sum()>0 ):
            lesion2[ind]=periventricular_idx

        #yuxtacortical
        elif( (votes==juxtacortical_idx).sum()>0 ):
            lesion2[ind]=juxtacortical_idx

        #cerebelar
        elif(  (votes==cerebelar_idx).sum()>0 ):
            lesion2[ind]=cerebelar_idx

        #yuxtacortical
        elif(  (votes==medular_idx).sum()>0 ):
            lesion2[ind]=medular_idx

        #deep
        else:
            lesion2[ind]=deepwhite_idx


    classified_name= fname_crisp.replace('crisp','lesions_types_')
    nii.Nifti1Image(lesion2, T1_img.affine).to_filename(classified_name)
    return classified_name, region_name, wm_name

def compute_volumes(im, labels, scale):
    assert(type(labels) is list)
    vols=[]
    for ll in labels:
        v = 0
        if not type(ll) is list:
            ll = [ll]
        for l in ll:
            a = (im == l)
            vl = np.sum(a[:])
            #print("l=", l, " -> volume=", vl)
            v += vl
        #print("==> ll=", ll, " -> total volume=", v)
        #vols.extend([(v*scale)/1000])
        vols.extend([(v*scale)])
    assert(len(vols) == len(labels))
    return vols

def readITKtransform( transform_file ):
    '''
    '''

    # read the transform
    transform = None
    with open( transform_file, 'r' ) as f:
        for line in f:

            # check for Parameters:
            if line.startswith( 'Parameters:' ):
                values = line.split( ': ' )[1].split( ' ' )

                # filter empty spaces and line breaks
                values = [float( e ) for e in values if ( e != '' and e != '\n' )]
                # create the upper left of the matrix
                transform_upper_left = np.reshape( values[0:9], ( 3, 3 ) )
                # grab the translation as well
                translation = values[9:]

            # check for FixedParameters:
            if line.startswith( 'FixedParameters:' ):
                values = line.split( ': ' )[1].split( ' ' )

                # filter empty spaces and line breaks
                values = [float( e ) for e in values if ( e != '' and e != '\n' )]
                # setup the center
                center = values

    # compute the offset
    offset = np.ones( 4 )
    for i in range( 0, 3 ):
        offset[i] = translation[i] + center[i];
        for j in range( 0, 3 ):
            offset[i] -= transform_upper_left[i][j] * center[i]

    # add the [0, 0, 0] line
    transform = np.vstack( ( transform_upper_left, [0, 0, 0] ) )
    # and the [offset, 1] column
    transform = np.hstack( ( transform, np.reshape( offset, ( 4, 1 ) ) ) )

    return transform

def get_expected_volumes(age, sex, tissue_vol, vol_ice):
    if(sex=='f' or sex=='femme' or sex=='woman' ):
        sex='female'
    if(sex=='m' or sex=='homme' or sex=='man' ):
        sex='male'
    structure=['White matter', 'Grey matter' ,'Cerebrospinal fluid']
    filenames=['WM.png','GM.png','CSF.png']
    dataset=load_obj('normal_crisp_volume_by_age')
    normal_vol=[]
    for i in range(3):
        if(sex=='uknown'):
            y1=(dataset['male'][i]['up']+dataset['female'][i]['up'])/2
            y2=(dataset['male'][i]['down']+dataset['female'][i]['down'])/2
        else:
            y1=dataset[sex][i]['up']
            y2=dataset[sex][i]['down']
        plt.fill_between(np.arange(101), y1, y2,color=['lightgreen'])
        plt.title(structure[i])
        plt.xlabel('age')
        plt.ylabel('volume (%)')
        if(not age=='uknown'):
            plt.plot([int(age)],[int(100*tissue_vol[i]/vol_ice)], 'ro')
            normal_vol.append([y2[int(age)],y1[int(age)]])
        plt.savefig(filenames[i], dpi=300)
        plt.clf()
    return filenames, normal_vol

def compute_SNR(ima, fima):
    assert(ima.shape == fima.shape)
    res = ima - fima
    level = filters.threshold_otsu(fima)
    ind = np.where(fima > level)
    noise = np.std(res[ind])
    return noise

def save_seg_nii(img, affine, input_filename, prefix):
    output_filename = get_filename(input_filename, prefix)
    OUT_TYPE=np.uint8
    assert(np.max(img) < np.iinfo(OUT_TYPE).max)
    OUT = img.astype(OUT_TYPE)
    array_img = nii.Nifti1Image(OUT, affine)
    array_img.set_data_dtype(OUT_TYPE)
    array_img.to_filename(output_filename)

def make_centered(im, width=256, height=256):
    assert(im.ndim == 3)
    assert(im.shape[0]<=width)
    assert(im.shape[1]<=height)
    y0 = int(height/2 - im.shape[0]/2)
    x0 = int(width/2 - im.shape[1]/2)
    assert(x0>=0 and x0<=width)
    assert(y0>=0 and y0<=height)
    out = np.zeros((height, width, 3), im.dtype)
    out[y0:y0+im.shape[0], x0:x0+im.shape[1], :] = im
    return out

def make_slice_image(T1_slice) :
    assert T1_slice.ndim == 2

    #Put values in [0; 255]
    im = T1_slice * 255.0

    #Add a channels dimension
    im = np.expand_dims(im, axis=-1)

    #Repeat value to have three-channel image
    im = np.tile(im, (1,1,3))

    out_im = im.astype(np.uint8)
    return out_im

def make_slice_with_seg_image_with_alpha_blending(T1_slice, LAB_slice, colors, alpha=0.8) :
    assert T1_slice.ndim == 2
    assert T1_slice.shape == LAB_slice.shape

    labels = list(np.unique(LAB_slice).astype(int))
    labels.remove(0) #remove background label
    maxLabel=np.max(labels)

    if (maxLabel >= len(colors)):
        print("ERROR: wrong number of colors")

    #Put values in [0; 255]
    im = T1_slice * 255.0

    # image premultiplied by 1-alpha
    aim = im * (1-alpha)

    #Add a channels dimension
    im = np.expand_dims(im, axis=-1)
    aim = np.expand_dims(aim, axis=-1)

    #Repeat value to have three-channel image
    im = np.tile(im, (1,1,3))
    aim = np.tile(aim, (1,1,3))

    acolors = colors * alpha

    for l in labels:
        im[LAB_slice == l] = aim[LAB_slice == l] + acolors[l, :]

    out_im = im.astype(np.uint8)
    return out_im

def get_patient_id(input_file):
    idStr = input_file.replace(".pdf", "") #get_filename(input_file, "", "")
    if len(idStr) > 20:
        idStr = idStr[0:14]+"..."
    return idStr

def get_color_string(color):
    return Template('\\mycbox{{rgb,255:red,$r;green,$g;blue,$b}}').safe_substitute(r=color[0], g=color[1], b=color[2])

def write_lesions(out, lesion_types_filename, scale):
    types=['healthy','Periventricular','Deepwhite','Juxtacortical','Cerebelar','Medular']
    lesion_mask = nii.load(lesion_types_filename).get_data()
    vol_tot = (compute_volumes((lesion_mask>0).astype('int'), [[1]], scale))[0]
    lesion_number=1
    for i in range(1,6):
        lesion_type= (lesion_mask==i).astype('int')
        seg_labels, seg_num = label(lesion_type, return_num=True, connectivity=2)
        if(seg_num>0):
            out.write(Template('\n').safe_substitute())
            out.write(Template('\\begin{tabularx}{0.9\\textwidth}{ X \centering{X} \centering{X} \centering{X} }\n').safe_substitute())
            out.write(Template(' \\cellcolor[gray]{0.9} {\\bfseries $t Lesions} & \\cellcolor[gray]{0.9} {\\bfseries Absolute vol. ($mm^{3}$)} & \\cellcolor[gray]{0.9} {\\bfseries Normalized vol. (\%)} & \\cellcolor[gray]{0.9} {\\bfseries Position (MNI coord.)}  \\\\\n').safe_substitute(t=types[i]))
            for j in range(1, seg_num+1):
                lesion= (seg_labels==j).astype('int')
                pos=center_of_mass(lesion)
                pos=[int(pos[0]),int(pos[1]),int(pos[2])]
                vol = (compute_volumes(lesion, [[1]], scale))[0]
                out.write(Template('Lesion $p & $g & $a & $d\\\\ \n').safe_substitute(p=lesion_number, g="{:5.2f}".format(vol), a="{:5.2f}".format(vol*100/vol_tot), d=pos))
                lesion_number=lesion_number+1
            out.write(Template('\\end{tabularx}\n').safe_substitute())
            out.write(Template('\n').safe_substitute())
            out.write(Template('\\vspace*{10pt}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\vspace*{10pt}\n').safe_substitute())

def write_lesion_table(out, lesion_types_filename,colors_lesions, scale):
    types=['healthy','Periventricular','Deepwhite','Juxtacortical','Cerebelar','Medular']
    lesion_mask = nii.load(lesion_types_filename).get_data()
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{ X \centering{X} \centering{X} \centering{X} }\n').safe_substitute())
    out.write(Template(' \\cellcolor[gray]{0.9} {\\bfseries Lesion Type } & \\cellcolor[gray]{0.9} {\\bfseries Count} &  \\cellcolor[gray]{0.9} {\\bfseries Absolute vol. ($mm^{3}$) }&\\cellcolor[gray]{0.9} {\\bfseries Normalized vol. (\%)}   \\\\\n').safe_substitute())
    lesion_type= (lesion_mask>0).astype('int')
    seg_labels, seg_num_tot = label(lesion_type, return_num=True, connectivity=2)
    vol_tot = (compute_volumes(lesion_type, [[1]], scale))[0]

    for i in range(1,6):
        cb=get_color_string(colors_lesions[i])
        lesion_type= (lesion_mask==i).astype('int')
        seg_labels, seg_num = label(lesion_type, return_num=True, connectivity=2)
        vol = (compute_volumes(lesion_type, [[1]], scale))[0]
        out.write(Template(' $cb  $p & $g & $a & $d\\\\ \n').safe_substitute(cb=cb, p=types[i], g=seg_num, a="{:5.2f}".format(vol), d="{:5.2f}".format(vol*100/vol_tot)))
    out.write(Template('\\hspace*{5pt} Total Lesions & $g & $a & $d\\\\ \n').safe_substitute(g=seg_num_tot, a="{:5.2f}".format(vol_tot), d="{:5.2f}".format(vol_tot*100/vol_tot)))
    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\vspace*{10pt}\n').safe_substitute())
    out.write(Template('\\pagebreak\n').safe_substitute())

def load_latex_packages(out):
    out.write(Template('\\documentclass[10pt,a4paper,oneside,notitlepage]{article}\n').safe_substitute())
    out.write(Template('\\usepackage{color}\n').safe_substitute())
    out.write(Template('\\usepackage[table,usenames,dvipsnames]{xcolor}\n').safe_substitute())
    out.write(Template('\\usepackage{mathptmx}\n').safe_substitute())
    out.write(Template('\\usepackage[T1]{fontenc}\n').safe_substitute())
    out.write(Template('\\usepackage[english]{babel}\n').safe_substitute())
    out.write(Template('\\usepackage{graphicx}\n').safe_substitute())
    out.write(Template('\\usepackage[cm]{fullpage}\n').safe_substitute())
    out.write(Template('\\usepackage{tabularx}\n').safe_substitute())
    out.write(Template('\\usepackage{array}\n').safe_substitute())
    out.write(Template('\\usepackage{multirow}\n').safe_substitute())
    out.write(Template('\\usepackage{subfig}\n').safe_substitute())
    out.write(Template('\\usepackage{tikz}\n').safe_substitute())
    out.write(Template('\\usepackage{hyperref}\n').safe_substitute())
    #out.write(Template('\newcolumntype{Y}{>{\centering\arraybackslash}X}').safe_substitute())
    out.write(Template('\n').safe_substitute())

def get_patient_info(out, basename,gender, age):
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X X X X}\n').safe_substitute())
    out.write(Template('\\cellcolor[gray]{0.9} {\\bfseries Patient ID} & \\cellcolor[gray]{0.9} {\\bfseries Sex} & \\cellcolor[gray]{0.9} {\\bfseries Age} & \\cellcolor[gray]{0.9} {\\bfseries Report Date} \\\\\n').safe_substitute())
    date = datetime.datetime.now().strftime("%d-%b-%Y")
    out.write(Template('$p & $g & $a & $d\\\\ \n').safe_substitute(p=basename, g=gender, a=age, d=date))
    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\vspace*{10pt}\n').safe_substitute())

def get_image_info(out, orientation_report, scale, snr):
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X X X X}\n').safe_substitute())
    out.write(Template('\\cellcolor[gray]{0.9} {\\bfseries Image information} & \\cellcolor[gray]{0.9} {\\bfseries Orientation} & \\cellcolor[gray]{0.9} {\\bfseries Scale factor} & \\cellcolor[gray]{0.9} {\\bfseries SNR} \\\\\n').safe_substitute())
    out.write(Template(' & $o & $sf & $snr\\\\ \n').safe_substitute(o=orientation_report, sf="{:5.2f}".format(scale), snr="{:5.2f}".format(snr)))
    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\vspace*{30pt}\n').safe_substitute())

def get_tissue_seg(out, vols_tissue,vol_ice, colors_ice, colors_tissue, normal_vol):
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{ X  \centering{X} \centering{X} }\n').safe_substitute())
    out.write(Template('\\cellcolor[gray]{0.96} {\\bfseries Tissues Segmentation} & \\cellcolor[gray]{0.96} {\\bfseries Absolute vol. ($mm^{3}$)} & \\cellcolor[gray]{0.96} {\\bfseries Normalized vol. (\%)}  \\\\\n').safe_substitute())
    vols_tissues_names=np.array(['healthy tissue', 'Lesions'])
    tissues_names=np.array(['Cerebrospinal fluid', 'Grey matter', 'White matter (including lesions)' ])
    cb=get_color_string(colors_ice[1])
    n="Intracranial Cavity (IC)"
    v=vol_ice
    p=100*v/vol_ice
    out.write(Template('$cb $n & $v & $p\\% \\\\\n').safe_substitute(n=n, cb=cb, v="{:5.2f}".format(v), p="{:5.2f}".format(p)))
    for i in range(len(tissues_names)):
        cb=get_color_string(colors_tissue[i+1])
        n=tissues_names[i]
        v=vols_tissue[i]
        p=100*v/vol_ice
        if(len(normal_vol)==0):
            out.write(Template('$cb $n & $v & $p\\% \\\\\n').safe_substitute(cb=cb, n=n, v="{:5.2f}".format(v), p="{:5.2f}".format(p)))
        else:
            out.write(Template('$cb $n & $v & $p\\% [$a - $b] \\\\\n').safe_substitute(cb=cb, n=n, v="{:5.2f}".format(v), p="{:5.2f}".format(p), a="{:5.2f}".format(normal_vol[i][0]), b="{:5.2f}".format(normal_vol[i][1])))

    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\vspace*{10pt}\n').safe_substitute())

def plot_img(out,plot_images_filenames):
    titles=['FLAIR','Intracranial cavity', 'Tissues', 'Lesions']
    for i in range(len(titles)):
        out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X}\n').safe_substitute())
        out.write(Template('\\cellcolor[gray]{0.9} {\\bfseries $v} \\\\\n').safe_substitute(v=titles[i]))
        out.write(Template('\\end{tabularx}\n').safe_substitute())
        out.write(Template('\\begin{tabularx}{0.8\\textwidth}{X}\n').safe_substitute())
        out.write(Template('\\centering \\includegraphics[width=0.25\\textwidth]{$f0} \\includegraphics[width=0.25\\textwidth]{$f1} \\includegraphics[width=0.25\\textwidth]{$f2}\\\\\n').safe_substitute(f0=plot_images_filenames[0,i], f1=plot_images_filenames[1,i], f2=plot_images_filenames[2,i]))
        out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\\pagebreak\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\vspace*{30pt}\n').safe_substitute())

def get_tissue_plot(out, filenames_normal_tissue):
    out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X}\n').safe_substitute())
    out.write(Template('\\cellcolor[gray]{0.9} {\\bfseries $v} \\\\\n').safe_substitute(v='Tissue expected volumes'))
    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\\begin{tabularx}{0.8\\textwidth}{X}\n').safe_substitute())
    out.write(Template('\\centering \\includegraphics[width=0.25\\textwidth]{$f0} \\includegraphics[width=0.25\\textwidth]{$f1} \\includegraphics[width=0.25\\textwidth]{$f2}\\\\\n').safe_substitute(f0=filenames_normal_tissue[0], f1=filenames_normal_tissue[1], f2=filenames_normal_tissue[2]))
    out.write(Template('\\end{tabularx}\n').safe_substitute())
    out.write(Template('\n').safe_substitute())
    out.write(Template('\\vspace*{30pt}\n').safe_substitute())

def save_pdf(input_file, age, gender, vols_tissue,vol_ice, snr, orientation_report,filenames_normal_tissue, normal_vol,
        scale,colors_ice, colors_lesions,colors_tissue,lesion_types_filename,
        plot_images_filenames):
    basename=os.path.basename(input_file).replace("preprocessed_mni_", "").replace("_t1", "").replace(".nii.gz", "")
    output_tex_filename = input_file.replace(".nii.gz", ".nii").replace(".nii", ".tex").replace("preprocessed_mni","report")
    print("output_tex_filename=", output_tex_filename)

    with open(output_tex_filename, 'w', newline='') as out:
        load_latex_packages(out)
        out.write(Template('\\newcommand{\\mycbox}[1]{\\tikz{\\path[fill=#1] (0.2,0.2) rectangle (0.4,0.4);}}\n').safe_substitute())
        out.write(Template('\n').safe_substitute())
        out.write(Template('\\newcolumntype{b}{>{\hsize=1.32\hsize}X}\n').safe_substitute())
        out.write(Template('\\newcolumntype{s}{>{\hsize=0.68\hsize}X}\n').safe_substitute())
        out.write(Template('\n').safe_substitute())
        out.write(Template('\\newcolumntype{B}{>{\hsize=1.8\hsize}X}\n').safe_substitute())
        out.write(Template('\\newcolumntype{R}{>{\hsize=0.8\hsize}X}\n').safe_substitute())
        out.write(Template('\\newcolumntype{S}{>{\hsize=0.9\hsize}X}\n').safe_substitute())
        out.write(Template('\\newcolumntype{T}{>{\hsize=0.6\hsize}X}\n').safe_substitute())
        out.write(Template('\n').safe_substitute())
        out.write(Template('\\hypersetup{colorlinks=true, urlcolor=magenta}').safe_substitute()) #allow to to not ahve a frame around the image
        out.write(Template('\n').safe_substitute())
        out.write(Template('\n').safe_substitute())
        out.write(Template('\\begin{document}\n').safe_substitute())
        out.write(Template('\\pagestyle{empty}\n').safe_substitute())
        filename_header = "header.png"
        out.write(Template('\\centering \\href{https://www.volbrain.net}{\\XeTeXLinkBox{ \\includegraphics[width=1\\textwidth]{$f}}}\\\\\n').safe_substitute(f=filename_header))
        out.write(Template('\\vspace*{20pt}\n').safe_substitute())
        out.write(Template('\\begin{tabularx}{0.9\\textwidth}{X X X X}\n').safe_substitute())
        out.write(Template('~~~~~~\\textcolor{NavyBlue}{version $v release $d}\n').safe_substitute(v=version, d=release_date))
        out.write(Template('\\end{tabularx}\n').safe_substitute())
        out.write(Template('\\begin{center}\n').safe_substitute())

        #Patient information
        get_patient_info(out, basename,gender, age)

        #Image information
        get_image_info(out, orientation_report, scale, snr)

        #Tissues Segmentation
        get_tissue_seg(out, vols_tissue,vol_ice, colors_ice, colors_tissue, normal_vol)

        #Tissue expected volumes
        get_tissue_plot(out, filenames_normal_tissue)

        #Lesion tables
        write_lesion_table(out, lesion_types_filename, colors_lesions, scale)

        #plot images
        plot_img(out, plot_images_filenames)

        #Lesion type tables
        write_lesions(out, lesion_types_filename, scale)

        #Footnotes
        out.write(Template('\\textcolor{blue}{\\footnotesize \\itshape *All the volumes are presented in absolute value (measured in $mm^{3}$) and in relative value (measured in relation to the IC volume).}\\\\*\n').safe_substitute())
        out.write(Template('\\textcolor{blue}{\\footnotesize \\itshape *Values between brackets show expected limits (95\%) of normalized volume in function of sex and age for each measure for reference purpose.}\\\\*\n').safe_substitute())
        out.write(Template('\\textcolor{blue}{\\footnotesize \\itshape *Position provides the $x$, $y$ and $z$ coordinates of the lesion center of mass.}\\\\*\n').safe_substitute())
        out.write(Template('\\textcolor{blue}{\\footnotesize \\itshape *Lesion burden is calculated as the lesion volume divided by the white matter volume.}\\\\*\n').safe_substitute())
        out.write(Template('\\textcolor{blue}{\\footnotesize \\itshape *Result images located in the MNI space (neurological orientation).}\\\\*\n').safe_substitute())
        out.write(Template('\\end{document}\n').safe_substitute())
        out.close()

        output_tex_basename=os.path.basename(output_tex_filename)
        output_tex_dirname=os.path.dirname(output_tex_filename)
        command="xelatex -interaction=nonstopmode -output-directory={} {}".format(output_tex_dirname, output_tex_basename)
        print(command)
        os.system(command)

        os.remove(output_tex_filename)
        os.remove(output_tex_filename.replace('tex', 'log'))
        os.remove(output_tex_filename.replace('tex', 'aux'))
        os.remove(output_tex_filename.replace('tex', 'out'))

def save_csv(input_file, vols):
    assert(len(vols)==len(labels_SLANT)+1)
    output_csv_filename = get_filename(input_file, "report_", ".csv")
    with open(output_csv_filename, mode='w') as output_file:
        csv_writer = csv.writer(output_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #write labels
        row = []
        for i in range(0, len(labels_SLANT)):
            row.extend([str(labels_SLANT[i])])
        row.extend(["mask"])
        csv_writer.writerow(row)
        #write labels names
        row = []
        for i in range(0, len(labels_SLANT_names)):
            row.extend([labels_SLANT_names[i]])
        row.extend(["mask"])
        csv_writer.writerow(row)
        #write values
        row = []
        for i in range(0, len(vols)):
            row.extend([str(vols[i])])
        csv_writer.writerow(row)

def save_images(suffixe, slice_index, FLAIR_slice,CRISP_slice,
                LAB_slice, MASK_slice, colors_ice,
                colors_lesions,colors_tissue,
                out_height=OUT_HEIGHT, alpha=DEFAULT_ALPHA):

    FLAIR_slice = np.rot90(FLAIR_slice)
    LAB_slice=np.rot90(LAB_slice)
    MASK_slice = np.rot90(MASK_slice)
    CRISP_slice=  np.rot90(CRISP_slice)

    out_im=make_slice_with_seg_image_with_alpha_blending(FLAIR_slice, LAB_slice, alpha=alpha, colors=colors_lesions)
    out_im = make_centered(out_im, out_height, out_height)
    filename_seg="seg_{}.png".format(suffixe)
    Image.fromarray(out_im, 'RGB').save(filename_seg)

    out_im=make_slice_with_seg_image_with_alpha_blending(FLAIR_slice, LAB_slice, alpha=0, colors=colors_lesions)
    out_im = make_centered(out_im, out_height, out_height)
    filename_flair="flair_{}.png".format(suffixe)
    Image.fromarray(out_im, 'RGB').save(filename_flair)

    out_im = make_slice_with_seg_image_with_alpha_blending(FLAIR_slice, MASK_slice, alpha=alpha, colors=colors_ice)
    out_im = make_centered(out_im, out_height, out_height)
    filename_ice="ice_{}.png".format(suffixe)
    Image.fromarray(out_im, 'RGB').save(filename_ice)

    out_im = make_slice_with_seg_image_with_alpha_blending(FLAIR_slice, CRISP_slice, alpha=alpha, colors=colors_tissue)
    out_im = make_centered(out_im, out_height, out_height)
    filename_tissue="tissue_{}.png".format(suffixe)
    Image.fromarray(out_im, 'RGB').save(filename_tissue)

    return filename_seg, filename_ice, filename_tissue, filename_flair
