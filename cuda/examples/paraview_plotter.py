# state file generated using paraview version 5.8.0

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# trace generated using paraview version 5.8.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [255, 727]
renderView1.InteractionMode = '2D'
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.06575775146484375, 0.010401725769042969, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [14.309981013419506, -5.277291586032522, 10000.0]
renderView1.CameraFocalPoint = [14.309981013419506, -5.277291586032522, 0.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 196.07503155558462
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1
renderView1.Background = [0, 0, 0]

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Active Brownian Particles #1')
layout1.AssignView(0, renderView1)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------
t0 = 0
tf = 1000
step = 10
folder = ''
file_name_prefix = "test_"
filenames_vec = [folder+file_name_prefix+'{:05d}.vtp'.format(t) for t in range(t0, tf, step)]
print(filenames_vec)
# create a new 'XML PolyData Reader'
MyReader = XMLPolyDataReader(FileName=filenames_vec)
MyReader.PointArrayStatus = ['id', 'director', 'velocity', 'force', 'radius']

# create a new 'Glyph'
glyph2 = Glyph(Input=MyReader,
    GlyphType='Sphere')
glyph2.OrientationArray = ['POINTS', 'No orientation array']
glyph2.ScaleArray = ['POINTS', 'radius']
glyph2.ScaleFactor = 2.0
glyph2.GlyphTransform = 'Transform2'
glyph2.GlyphMode = 'All Points'

# create a new 'Glyph'
glyph1 = Glyph(Input=MyReader,
    GlyphType='Arrow')
glyph1.OrientationArray = ['POINTS', 'director']
glyph1.ScaleArray = ['POINTS', 'director']
glyph1.ScaleFactor = 2.0
glyph1.GlyphTransform = 'Transform2'
glyph1.GlyphMode = 'All Points'

# init the 'Arrow' selected for 'GlyphType'
glyph1.GlyphType.TipResolution = 1
glyph1.GlyphType.TipLength = 0.4
glyph1.GlyphType.ShaftResolution = 4
glyph1.GlyphType.ShaftRadius = 0.02

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from MyReader
MyReaderDisplay = Show(MyReader, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
MyReaderDisplay.Representation = 'Surface'
MyReaderDisplay.ColorArrayName = [None, '']
MyReaderDisplay.OSPRayScaleArray = 'director'
MyReaderDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
MyReaderDisplay.SelectOrientationVectors = 'None'
MyReaderDisplay.ScaleFactor = 4.990595817565918
MyReaderDisplay.SelectScaleArray = 'None'
MyReaderDisplay.GlyphType = 'Arrow'
MyReaderDisplay.GlyphTableIndexArray = 'None'
MyReaderDisplay.GaussianRadius = 0.2495297908782959
MyReaderDisplay.SetScaleArray = ['POINTS', 'director']
MyReaderDisplay.ScaleTransferFunction = 'PiecewiseFunction'
MyReaderDisplay.OpacityArray = ['POINTS', 'director']
MyReaderDisplay.OpacityTransferFunction = 'PiecewiseFunction'
MyReaderDisplay.DataAxesGrid = 'GridAxesRepresentation'
MyReaderDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
MyReaderDisplay.ScaleTransferFunction.Points = [-0.9999980551566605, 0.0, 0.5, 0.0, 0.9999712702481867, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
MyReaderDisplay.OpacityTransferFunction.Points = [-0.9999980551566605, 0.0, 0.5, 0.0, 0.9999712702481867, 1.0, 0.5, 0.0]

# show data from glyph1
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.AmbientColor = [1.0, 0.0, 0.0]
glyph1Display.ColorArrayName = [None, '']
glyph1Display.DiffuseColor = [1.0, 0.0, 0.0]
glyph1Display.OSPRayScaleArray = 'director'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'None'
glyph1Display.ScaleFactor = 5.138651657104493
glyph1Display.SelectScaleArray = 'None'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.GlyphTableIndexArray = 'None'
glyph1Display.GaussianRadius = 0.25693258285522463
glyph1Display.SetScaleArray = ['POINTS', 'director']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph1Display.OpacityArray = ['POINTS', 'director']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph1Display.ScaleTransferFunction.Points = [-0.9999980551566605, 0.0, 0.5, 0.0, 0.9999712702481867, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph1Display.OpacityTransferFunction.Points = [-0.9999980551566605, 0.0, 0.5, 0.0, 0.9999712702481867, 1.0, 0.5, 0.0]

# show data from glyph2
glyph2Display = Show(glyph2, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
glyph2Display.Representation = 'Surface'
glyph2Display.ColorArrayName = [None, '']
glyph2Display.OSPRayScaleArray = 'Normals'
glyph2Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph2Display.SelectOrientationVectors = 'None'
glyph2Display.ScaleFactor = 10.04793348312378
glyph2Display.SelectScaleArray = 'None'
glyph2Display.GlyphType = 'Arrow'
glyph2Display.GlyphTableIndexArray = 'None'
glyph2Display.GaussianRadius = 0.5023966741561889
glyph2Display.SetScaleArray = ['POINTS', 'Normals']
glyph2Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph2Display.OpacityArray = ['POINTS', 'Normals']
glyph2Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph2Display.DataAxesGrid = 'GridAxesRepresentation'
glyph2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph2Display.ScaleTransferFunction.Points = [-0.9749279618263245, 0.0, 0.5, 0.0, 0.9749279618263245, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph2Display.OpacityTransferFunction.Points = [-0.9749279618263245, 0.0, 0.5, 0.0, 0.9749279618263245, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(MyReader)
# ----------------------------------------------------------------